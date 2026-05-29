import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, 
  Upload, 
  Trash2, 
  Leaf, 
  Info, 
  CheckCircle, 
  Activity, 
  ShieldAlert, 
  TrendingUp, 
  RefreshCw
} from 'lucide-react';

interface PredictionResponse {
  prediction: string;
  confidence: number;
  label: string;
  recyclable: boolean;
  bin: string;
  instructions: string[];
  impact: string;
  latency_seconds?: number;
  is_mocked?: boolean;
}

interface HistoryItem {
  category: string;
  label: string;
  timestamp: string;
  recyclable: boolean;
}

const DATASET_DISTRIBUTION = [
  { name: "Vegetation", count: 976, percentage: "20.5%" },
  { name: "Plastic", count: 921, percentage: "19.4%" },
  { name: "Paper", count: 777, percentage: "16.4%" },
  { name: "Food Organics", count: 650, percentage: "13.7%" },
  { name: "Cardboard", count: 461, percentage: "9.7%" },
  { name: "Glass", count: 418, percentage: "8.8%" },
  { name: "Metal", count: 320, percentage: "6.7%" },
  { name: "Miscellaneous Trash", count: 213, percentage: "4.5%" },
  { name: "Textile Trash", count: 16, percentage: "0.3%" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'camera'>('camera');
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'sandbox'>('checking');
  const [modelType, setModelType] = useState<string>('');
  
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isLiveActive, setIsLiveActive] = useState<boolean>(true);
  const isLiveActiveRef = useRef(true);
  const requestInProgressRef = useRef(false);
  const activeLoopRef = useRef(false);

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [scanHistory, setScanHistory] = useState<HistoryItem[]>([]);

  const checkBackend = async () => {
    try {
      setBackendStatus('checking');
      const response = await fetch('/api/health');
      if (response.ok) {
        const data = await response.json();
        setBackendStatus('online');
        setModelType(data.model_loaded ? 'ResNet18 Fine-Tuned Loaded' : 'Simulated Sandbox Mode');
      } else {
        setBackendStatus('sandbox');
        setModelType('Client Sandbox Inference');
      }
    } catch (err) {
      setBackendStatus('sandbox');
      setModelType('Client Sandbox Inference');
    }
  };

  useEffect(() => {
    checkBackend();
    const saved = localStorage.getItem('ecosort_scans');
    if (saved) {
      try {
        setScanHistory(JSON.parse(saved));
      } catch (e) {}
    }
  }, []);

  useEffect(() => {
    isLiveActiveRef.current = isLiveActive;
  }, [isLiveActive]);

  const runLiveDetectionLoop = async () => {
    if (!activeLoopRef.current || !videoRef.current) return;

    const video = videoRef.current;
    if (
      video.readyState === video.HAVE_ENOUGH_DATA && 
      isLiveActiveRef.current && 
      !requestInProgressRef.current
    ) {
      requestInProgressRef.current = true;

      const canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 240;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(async (blob) => {
          if (blob && activeLoopRef.current && isLiveActiveRef.current) {
            const file = new File([blob], `live_frame.jpg`, { type: 'image/jpeg' });
            try {
              await classifyImage(file, true);
            } catch (e) {
              console.error("Live detection loop error:", e);
            }
          }
          requestInProgressRef.current = false;
          
          if (activeLoopRef.current) {
            setTimeout(runLiveDetectionLoop, 400);
          }
        }, 'image/jpeg', 0.6);
      } else {
        requestInProgressRef.current = false;
        if (activeLoopRef.current) {
          setTimeout(runLiveDetectionLoop, 400);
        }
      }
    } else {
      if (activeLoopRef.current) {
        setTimeout(runLiveDetectionLoop, 200);
      }
    }
  };

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      activeLoopRef.current = true;
      setTimeout(runLiveDetectionLoop, 800);
    } catch (err: any) {
      console.error("Camera access failed:", err);
      setCameraError("Unable to access camera. Please check camera permissions.");
      setActiveTab('upload');
    }
  };

  const stopCamera = () => {
    activeLoopRef.current = false;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  useEffect(() => {
    if (activeTab === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [activeTab]);

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
    }
  };

  const captureSnapshot = () => {
    if (videoRef.current) {
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], `webcam_snap_${Date.now()}.jpg`, { type: 'image/jpeg' });
            setSelectedFile(file);
            setPreviewUrl(canvas.toDataURL('image/jpeg'));
            stopCamera();
            classifyImage(file);
          }
        }, 'image/jpeg');
      }
    }
  };

  const classifyImage = async (fileToUpload?: File, isLive = false) => {
    const targetFile = fileToUpload || selectedFile;
    if (!targetFile) return;

    if (!isLive) {
      setIsLoading(true);
      setPrediction(null);
    }

    const formData = new FormData();
    formData.append('file', targetFile);

    try {
      let resultData: PredictionResponse;

      if (backendStatus === 'online') {
        const response = await fetch('/api/classify', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Inference server error.');
        }
        resultData = await response.json();
      } else {
        if (!isLive) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
        } else {
          await new Promise((resolve) => setTimeout(resolve, 300));
        }
        
        const filename = targetFile.name.toLowerCase();
        let predictionKey = "plastic";
        
        if (filename.includes("plastic") || filename.includes("bottle") || filename.includes("snap")) {
          predictionKey = "plastic";
        } else if (filename.includes("can") || filename.includes("metal") || filename.includes("tin") || filename.includes("soda")) {
          predictionKey = "metal";
        } else if (filename.includes("card") || filename.includes("box") || filename.includes("package")) {
          predictionKey = "cardboard";
        } else if (filename.includes("glass") || filename.includes("jar") || filename.includes("bottle")) {
          predictionKey = "glass";
        } else if (filename.includes("paper") || filename.includes("news") || filename.includes("book")) {
          predictionKey = "paper";
        } else if (filename.includes("food") || filename.includes("apple") || filename.includes("banana") || filename.includes("scrap")) {
          predictionKey = "food organics";
        } else if (filename.includes("leave") || filename.includes("grass") || filename.includes("plant") || filename.includes("branch")) {
          predictionKey = "vegetation";
        } else if (filename.includes("cloth") || filename.includes("shirt") || filename.includes("fabric")) {
          predictionKey = "textile trash";
        } else {
          const keys = ["vegetation", "plastic", "paper", "food organics", "cardboard", "glass", "metal", "miscellaneous trash"];
          predictionKey = keys[Math.floor(Math.random() * keys.length)];
        }

        const mockMetaData: Record<string, any> = {
          "cardboard": {
            "title": "Cardboard",
            "recyclable": true,
            "bin": "Recycle Bin (Blue)",
            "instructions": [
              "Flatten boxes completely to save bin space.",
              "Remove all plastic packaging, packing peanuts, and excessive shipping tape.",
              "Ensure the cardboard is dry and free of food grease."
            ],
            "impact": "Recycling cardboard saves 24% of the energy needed to make new cardboard."
          },
          "food organics": {
            "title": "Food Organics",
            "recyclable": true,
            "bin": "Compost Bin (Green)",
            "instructions": [
              "Place fruit peels, vegetable scraps, coffee grounds, and leftovers here.",
              "Ensure no plastic wrappers or metal twist ties are attached.",
              "Animal bones and dairy can go in municipal green bins."
            ],
            "impact": "Composting food waste prevents methane emissions from decomposing landfill waste."
          },
          "glass": {
            "title": "Glass Bottles & Jars",
            "recyclable": true,
            "bin": "Glass Bin / Recycle Bin (Blue)",
            "instructions": [
              "Empty and rinse the glass jar or bottle thoroughly.",
              "Remove metal lids (they are recyclable separately).",
              "Do not put drinking glasses, window pane glass, or ceramics here."
            ],
            "impact": "Glass can be recycled infinitely without losing purity or quality."
          },
          "metal": {
            "title": "Metal & Cans",
            "recyclable": true,
            "bin": "Recycle Bin (Blue)",
            "instructions": [
              "Empty and rinse aluminum beverage cans and steel food cans.",
              "Crumpled aluminum foil is recyclable if it is clean and balled up.",
              "Crush cans to maximize bin space."
            ],
            "impact": "Recycling metal saves up to 95% of the energy needed to extract new raw material."
          },
          "miscellaneous trash": {
            "title": "General Trash / Landfill",
            "recyclable": false,
            "bin": "Landfill Bin (Black/Gray)",
            "instructions": [
              "Used for non-recyclable materials like diapers, multi-layered chip bags, and styrofoam.",
              "Ensure items are bagged securely.",
              "Consider reducing the purchase of single-use items in this category."
            ],
            "impact": "Items in this bin will be buried in landfills. Reducing this waste is the highest priority."
          },
          "paper": {
            "title": "Mixed Paper",
            "recyclable": true,
            "bin": "Recycle Bin (Blue)",
            "instructions": [
              "Includes office paper, newspaper, magazines, and clean paper bags.",
              "Shredded paper should be placed in a paper bag before recycling.",
              "Ensure paper is clean and dry. Wet paper fibers clog machinery."
            ],
            "impact": "Recycling one ton of paper saves 17 trees and 7,000 gallons of water."
          },
          "plastic": {
            "title": "Recyclable Plastics",
            "recyclable": true,
            "bin": "Recycle Bin (Blue)",
            "instructions": [
              "Focus on clean plastic bottles, jugs, tubs, and jars (PET #1 and HDPE #2).",
              "Rinse out food and liquid residues. Squashing helps save space.",
              "Discard thin plastic bags and wraps."
            ],
            "impact": "Plastics take up to 500 years to decompose. Recycling them keeps microplastics out of the ocean."
          },
          "textile trash": {
            "title": "Textile Trash / Clothing",
            "recyclable": false,
            "bin": "Donation / Fabric Recycling",
            "instructions": [
              "Wearable clothes and shoes should be donated to charities.",
              "Unusable, torn, or stained clothes can be cut up into cleaning rags.",
              "Look for specialized textile recycling drop-off centers."
            ],
            "impact": "Nearly 85% of textiles end up in landfills. Donation extends their lifecycle."
          },
          "vegetation": {
            "title": "Yard Waste / Vegetation",
            "recyclable": true,
            "bin": "Yard Waste Bin (Green)",
            "instructions": [
              "Includes leaves, grass clippings, weeds, branches, and garden trimmings.",
              "Do not put treated wood, large logs, or rocks in this bin.",
              "Never place yard waste in plastic garbage bags; use paper yard bags."
            ],
            "impact": "Yard waste is turned into high-quality mulch and soil conditioners."
          }
        };

        const meta = mockMetaData[predictionKey] || mockMetaData["miscellaneous trash"];
        resultData = {
          prediction: predictionKey,
          confidence: Number((0.85 + Math.random() * 0.14).toFixed(4)),
          label: meta.title,
          recyclable: meta.recyclable,
          bin: meta.bin,
          instructions: meta.instructions,
          impact: meta.impact,
          latency_seconds: 0.25,
          is_mocked: true
        };
      }

      setPrediction(resultData);

      const historyItem: HistoryItem = {
        category: resultData.prediction,
        label: resultData.label,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        recyclable: resultData.recyclable
      };
      
      const newHistory = [historyItem, ...scanHistory].slice(0, 100);
      setScanHistory(newHistory);
      localStorage.setItem('ecosort_scans', JSON.stringify(newHistory));

    } catch (err) {
      console.error(err);
      if (!isLive) {
        setPrediction({
          prediction: "miscellaneous trash",
          confidence: 0.5,
          label: "General Trash / Landfill",
          recyclable: false,
          bin: "Landfill Bin (Black/Gray)",
          instructions: ["Inference service is currently busy or offline.", "Dispose in general trash if uncertain."],
          impact: "Diverting waste reduces greenhouse gases. Check local rules.",
          is_mocked: true
        });
      }
    } finally {
      if (!isLive) {
        setIsLoading(false);
      }
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    if (activeTab === 'camera') {
      startCamera();
    }
  };

  const totalScans = scanHistory.length;
  const recyclableCount = scanHistory.filter(item => item.recyclable).length;
  const compostableCount = scanHistory.filter(item => item.category === 'food organics' || item.category === 'vegetation').length;
  const recyclingRate = totalScans > 0 
    ? Math.round(((recyclableCount + compostableCount) / totalScans) * 100) 
    : 0;

  const calculateDashOffset = (confidence: number) => {
    const radius = 28;
    const circumference = 2 * Math.PI * radius;
    return circumference - (confidence * circumference);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo">
            <Leaf size={32} strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="brand-title">EcoSort AI</h1>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', display: 'block', marginTop: '-2px' }}>
              Real-World Garbage Model Sorting Dashboard
            </span>
          </div>
        </div>

        <div className="system-status">
          <div className={`status-dot ${backendStatus === 'sandbox' ? 'simulated' : ''}`}></div>
          <span>
            {backendStatus === 'checking' && 'Connecting...'}
            {backendStatus === 'online' && `System Active (${modelType})`}
            {backendStatus === 'sandbox' && `Local Sandbox (${modelType})`}
          </span>
          <button 
            onClick={checkBackend} 
            title="Reconnect backend" 
            style={{ background: 'transparent', border: 'none', color: 'var(--color-text-secondary)', cursor: 'pointer', display: 'flex', marginLeft: '6px' }}
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </header>

      <div className="dashboard-grid">
        <section className="glass-card scanner-container" style={{ padding: '24px' }}>
          <h2 style={{ fontSize: '1.25rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={20} color="var(--color-primary)" />
            Sorting Scanner Console
          </h2>
          
          <nav className="tab-nav">
            <button 
              className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              <Upload size={16} />
              Upload Image
            </button>
            <button 
              className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`}
              onClick={() => setActiveTab('camera')}
            >
              <Camera size={16} />
              Live Scanner
            </button>
          </nav>

          <div 
            className={`scanner-area ${isDragOver ? 'drag-over' : ''}`}
            onDragOver={handleDragEnter}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {activeTab === 'upload' ? (
              !previewUrl ? (
                <label className="upload-prompt">
                  <input 
                    type="file" 
                    className="file-input" 
                    accept="image/*" 
                    onChange={handleFileChange} 
                  />
                  <div className="upload-icon-container">
                    <Upload size={36} />
                  </div>
                  <div>
                    <p style={{ fontWeight: 600, fontSize: '1.05rem', color: '#fff' }}>Drag and drop trash image here</p>
                    <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '4px' }}>
                      Supports PNG, JPG, JPEG up to 10MB
                    </p>
                  </div>
                  <span className="btn-primary" style={{ marginTop: '8px' }}>Browse Files</span>
                </label>
              ) : (
                <div className="preview-container">
                  <button className="clear-btn" onClick={clearSelection} title="Clear Image">
                    <Trash2 size={18} />
                  </button>
                  <img src={previewUrl} alt="Trash preview" className="preview-image" />
                  
                  {!prediction && !isLoading && (
                    <button 
                      className="btn-primary" 
                      style={{ marginTop: '20px', padding: '12px 32px' }}
                      onClick={() => classifyImage()}
                    >
                      Classify Material
                    </button>
                  )}
                </div>
              )
            ) : (
              <div className="webcam-wrapper">
                {cameraError ? (
                  <div style={{ textAlign: 'center', color: 'var(--color-danger)', padding: '20px' }}>
                    <ShieldAlert size={40} style={{ marginBottom: '12px' }} />
                    <p>{cameraError}</p>
                    <button 
                      className="btn-primary" 
                      style={{ marginTop: '16px' }} 
                      onClick={() => setActiveTab('upload')}
                    >
                      Back to Upload
                    </button>
                  </div>
                ) : (
                  <>
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline 
                      muted 
                      className="webcam-video"
                    />
                    <div className="webcam-controls" style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                      <button 
                        className="btn-primary" 
                        onClick={() => setIsLiveActive(!isLiveActive)}
                        style={{
                          backgroundColor: isLiveActive ? 'var(--color-primary)' : 'rgba(255,255,255,0.05)',
                          border: isLiveActive ? 'none' : '1px solid var(--border-color)',
                          color: '#fff',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px'
                        }}
                      >
                        <RefreshCw size={16} className={isLiveActive ? 'spin-slow' : ''} style={{ animation: isLiveActive ? 'spin 3s linear infinite' : 'none' }} />
                        {isLiveActive ? 'Live Scanning Active' : 'Resume Live Scanning'}
                      </button>
                      
                      {!isLiveActive && (
                        <button className="btn-primary" onClick={captureSnapshot} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <Camera size={16} />
                          Scan Single Frame
                        </button>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </section>

        <section className="glass-card results-card">
          {isLoading ? (
            <div className="empty-results">
              <div className="spinner"></div>
              <h3 style={{ color: '#fff', fontSize: '1.1rem', fontWeight: 600 }}>Analyzing Waste Texture...</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '6px' }}>
                Matching against deformed and degraded material profiles
              </p>
            </div>
          ) : prediction ? (
            <div className="fade-in">
              <div className="results-header">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span className={`category-badge ${prediction.recyclable ? 'recyclable-badge' : 'landfill-badge'}`}>
                      {prediction.recyclable ? 'Recyclable' : 'Non-Recyclable'}
                    </span>
                    {isLiveActive && activeTab === 'camera' && (
                      <span className="category-badge" style={{ background: 'rgba(16, 185, 129, 0.1)', color: 'var(--color-primary)', border: '1px dashed var(--color-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span className="status-dot" style={{ width: '6px', height: '6px', margin: 0 }}></span>
                        LIVE
                      </span>
                    )}
                  </div>
                  <h3 className="prediction-title">{prediction.label}</h3>
                </div>
                
                <div className="confidence-section">
                  <div className="gauge-container">
                    <svg className="gauge-svg">
                      <circle cx="35" cy="35" r="28" className="gauge-bg" />
                      <circle 
                        cx="35" 
                        cy="35" 
                        r="28" 
                        className="gauge-value" 
                        strokeDasharray={2 * Math.PI * 28}
                        strokeDashoffset={calculateDashOffset(prediction.confidence)}
                        style={{
                          stroke: prediction.confidence > 0.8 
                            ? 'var(--color-primary)' 
                            : prediction.confidence > 0.6 
                            ? 'var(--color-warning)' 
                            : 'var(--color-danger)'
                        }}
                      />
                    </svg>
                    <div className="gauge-text">
                      {Math.round(prediction.confidence * 100)}%
                    </div>
                  </div>
                </div>
              </div>

              <div className="destination-bin">
                <div 
                  className="bin-swatch"
                  style={{
                    backgroundColor: prediction.bin.includes("Blue") 
                      ? 'var(--bin-recycle)' 
                      : prediction.bin.includes("Green") 
                      ? 'var(--bin-compost)' 
                      : prediction.bin.includes("Glass") 
                      ? 'var(--bin-glass)' 
                      : prediction.bin.includes("Donation") 
                      ? 'var(--bin-special)' 
                      : 'var(--bin-landfill)'
                  }}
                />
                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', display: 'block', textTransform: 'uppercase' }}>
                    Separation Route
                  </span>
                  <strong style={{ color: '#fff', fontSize: '1.05rem' }}>{prediction.bin}</strong>
                </div>
              </div>

              <div className="instructions-section">
                <h4 className="instructions-title">EcoSort Sorting Directives:</h4>
                <ul className="instructions-list">
                  {prediction.instructions.map((inst, index) => (
                    <li key={index}>{inst}</li>
                  ))}
                </ul>
              </div>

              <div className="impact-callout">
                <p style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                  <Info size={16} style={{ flexShrink: 0, marginTop: '2px', color: 'var(--color-primary)' }} />
                  <span>{prediction.impact}</span>
                </p>
              </div>
              
              <div style={{ marginTop: '24px', fontSize: '0.75rem', color: 'var(--color-text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Inference time: {prediction.latency_seconds || '0.25'}s</span>
                {prediction.is_mocked && <span>(Sandbox Simulation Mode)</span>}
              </div>
            </div>
          ) : (
            <div className="empty-results">
              {activeTab === 'camera' && isLiveActive ? (
                <>
                  <div className="status-dot" style={{ width: '12px', height: '12px', marginBottom: '16px', backgroundColor: 'var(--color-primary)' }}></div>
                  <h3>Live Scan Running</h3>
                  <p style={{ fontSize: '0.875rem', maxWidth: '280px', marginTop: '6px' }}>
                    Position a waste item (crushed bottle, crumpled paper, can) in front of the camera to detect its category.
                  </p>
                </>
              ) : (
                <>
                  <Camera size={48} style={{ opacity: 0.15, marginBottom: '16px' }} />
                  <h3>Awaiting Input</h3>
                  <p style={{ fontSize: '0.875rem', maxWidth: '280px', marginTop: '6px' }}>
                    Upload a garbage photo or capture it live with the camera scanner to classify and sort.
                  </p>
                </>
              )}
            </div>
          )}
        </section>

        <div className="stats-row">
          <div className="glass-card stat-card">
            <div className="stat-icon">
              <Activity size={24} />
            </div>
            <div>
              <div className="stat-number">{totalScans}</div>
              <div className="stat-label">Total Scans</div>
            </div>
          </div>
          
          <div className="glass-card stat-card">
            <div className="stat-icon">
              <CheckCircle size={24} />
            </div>
            <div>
              <div className="stat-number">{recyclableCount}</div>
              <div className="stat-label">Recyclables</div>
            </div>
          </div>

          <div className="glass-card stat-card">
            <div className="stat-icon">
              <Leaf size={24} />
            </div>
            <div>
              <div className="stat-number">{compostableCount}</div>
              <div className="stat-label">Compostables</div>
            </div>
          </div>

          <div className="glass-card stat-card">
            <div className="stat-icon">
              <TrendingUp size={24} />
            </div>
            <div>
              <div className="stat-number">{recyclingRate}%</div>
              <div className="stat-label">Landfill Diverted</div>
            </div>
          </div>
        </div>

        <div className="info-section">
          <section className="glass-card edu-card">
            <h3 className="edu-title">
              <Info size={20} color="var(--color-primary)" />
              Why Real-World Waste Datasets Matter
            </h3>
            
            <div className="edu-grid">
              <div className="edu-item">
                <h4>Traditional Clean Datasets</h4>
                <p style={{ marginTop: '8px' }}>
                  Traditional sorting models are trained on clean, isolated product photos (e.g., straight plastic bottles on solid white backgrounds). 
                  In practice, these models fail because real garbage is degraded, crumpled, dirty, and blended together.
                </p>
              </div>
              
              <div className="edu-item">
                <h4>Conveyor Landfill Datasets (RealWaste)</h4>
                <p style={{ marginTop: '8px' }}>
                  Our system is built around the <strong>RealWaste</strong> dataset. Items were photographed at an active landfill site. 
                  By training on crumpled cans, dirty papers, crushed bottles, and organic decay, our model achieves high real-world accuracy in sorting facilities.
                </p>
              </div>
            </div>
            
            <div style={{ marginTop: '20px', borderTop: '1px solid var(--border-color)', paddingTop: '16px', fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
              <strong>Dataset reference:</strong> Knoblauch et al., <i>RealWaste</i>. Sourced from the municipal waste streams, 
              providing authentic visual representations of trash. Learn more at the <a href="https://archive.ics.uci.edu/dataset/908/realwaste" target="_blank" rel="noreferrer" style={{ color: 'var(--color-primary)', textDecoration: 'underline' }}>UCI Repository</a>.
            </div>
          </section>

          <section className="glass-card chart-card">
            <h3 className="edu-title" style={{ marginBottom: '8px' }}>
              <TrendingUp size={20} color="var(--color-primary)" />
              RealWaste Class Profiles
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
              Distribution of the 4,752 images across categories in the training pool.
            </p>
            
            <div className="chart-container">
              {DATASET_DISTRIBUTION.map((item, index) => (
                <div key={index} className="chart-bar-wrapper">
                  <div className="chart-bar-labels">
                    <span>{item.name}</span>
                    <span>{item.count} ({item.percentage})</span>
                  </div>
                  <div className="chart-bar-track">
                    <div 
                      className="chart-bar-fill" 
                      style={{ 
                        width: item.percentage,
                        backgroundColor: item.name === 'Miscellaneous Trash' 
                          ? 'var(--bin-landfill)' 
                          : item.name === 'Plastic' || item.name === 'Metal' || item.name === 'Paper' || item.name === 'Cardboard'
                          ? 'var(--bin-recycle)'
                          : item.name === 'Food Organics' || item.name === 'Vegetation'
                          ? 'var(--bin-compost)'
                          : 'var(--color-primary)'
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>

      <footer className="app-footer">
        <p>EcoSort AI Waste Classifier — Built with React, Vite & PyTorch ResNet18</p>
        <p style={{ marginTop: '6px', fontSize: '0.8rem' }}>
          Open Source project designed for sorting facility automation. Dataset: <a href="https://www.kaggle.com/datasets/kneroma/realwaste" target="_blank" rel="noreferrer">RealWaste Kaggle</a>.
        </p>
      </footer>
    </div>
  );
}
