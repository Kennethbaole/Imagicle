import React, { useState, useRef, useEffect } from "react";
import { Play, Download, X, RotateCcw, Maximize2 } from "lucide-react";
import * as THREE from "three";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
// add this near the top of the file:
console.log("VITE_API_BASE =", API_BASE);
// optional: expose to console for quick checks
window.API_BASE = API_BASE;


const ImageGeneratorPage = () => {
  const [prompt, setPrompt] = useState("");
  const [plyUrl, setPlyUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [showModelModal, setShowModelModal] = useState(false);
  const mountRef = useRef(null);
  const modalMountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const modalSceneRef = useRef(null);
  const modalRendererRef = useRef(null);

  const handleGenerate = async () => {
    if (!prompt) return;
    setLoading(true);
    setPlyUrl("");
    setShowModelModal(false);

    try {
      // CHANGED: use API_BASE and show backend error body
      const response = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text}`);
      }

      const data = await response.json();
      console.log("API Response:", data);
      setPlyUrl(data.url);
      setTimeout(() => setShowModelModal(true), 500);
    } catch (err) {
      console.error(err);
      alert(`Error generating 3D model:\n${err.message ?? err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = plyUrl;
    link.download = `generated-3d-model-${Date.now()}.ply`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Simple PLY parser for basic geometry
  const parsePLY = async (url) => {
    try {
      const response = await fetch(url);
      const text = await response.text();
      const lines = text.split('\n');
      
      let vertexCount = 0;
      let faceCount = 0;
      let headerEnded = false;
      const vertices = [];
      const faces = [];
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        if (line === 'end_header') {
          headerEnded = true;
          continue;
        }
        
        if (!headerEnded) {
          if (line.startsWith('element vertex')) {
            vertexCount = parseInt(line.split(' ')[2]);
          } else if (line.startsWith('element face')) {
            faceCount = parseInt(line.split(' ')[2]);
          }
          continue;
        }
        
        if (vertices.length < vertexCount) {
          const coords = line.split(' ').map(parseFloat);
          if (coords.length >= 3) {
            vertices.push(coords[0], coords[1], coords[2]);
          }
        } else if (faces.length < faceCount * 4) {
          const faceData = line.split(' ').map(parseInt);
          if (faceData.length >= 4 && faceData[0] === 3) {
            faces.push(faceData[1], faceData[2], faceData[3]);
          }
        }
      }
      
      return { vertices: new Float32Array(vertices), indices: new Uint32Array(faces) };
    } catch (error) {
      console.error('Error parsing PLY file:', error);
      return null;
    }
  };

  const create3DViewer = async (mountElement, plyUrl, isModal = false) => {
    if (!mountElement || !plyUrl) return null;

    // Clear previous content
    while (mountElement.firstChild) {
      mountElement.removeChild(mountElement.firstChild);
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    const camera = new THREE.PerspectiveCamera(75, mountElement.clientWidth / mountElement.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountElement.clientWidth, mountElement.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountElement.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Load and display PLY model
    try {
      const plyData = await parsePLY(plyUrl);
      if (plyData) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(plyData.vertices, 3));
        if (plyData.indices.length > 0) {
          geometry.setIndex(new THREE.BufferAttribute(plyData.indices, 1));
        }
        geometry.computeVertexNormals();

        const material = new THREE.MeshPhongMaterial({ 
          color: 0x00ff88,
          shininess: 100,
          transparent: true,
          opacity: 0.9
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);

        // Center and scale the model
        const box = new THREE.Box3().setFromObject(mesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxSize = Math.max(size.x, size.y, size.z);
        
        mesh.position.sub(center);
        mesh.scale.multiplyScalar(2 / maxSize);
        
        camera.position.set(3, 3, 3);
        camera.lookAt(0, 0, 0);
      }
    } catch (error) {
      console.error('Error loading PLY model:', error);
      // Fallback: show a simple cube
      const geometry = new THREE.BoxGeometry();
      const material = new THREE.MeshPhongMaterial({ color: 0x00ff88 });
      const cube = new THREE.Mesh(geometry, material);
      scene.add(cube);
      camera.position.z = 3;
    }

    // Mouse controls for rotation
    let isMouseDown = false;
    let mouseX = 0;
    let mouseY = 0;
    let targetRotationX = 0;
    let targetRotationY = 0;

    const onMouseDown = (event) => {
      isMouseDown = true;
      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const onMouseMove = (event) => {
      if (isMouseDown) {
        targetRotationY += (event.clientX - mouseX) * 0.01;
        targetRotationX += (event.clientY - mouseY) * 0.01;
        mouseX = event.clientX;
        mouseY = event.clientY;
      }
    };

    const onMouseUp = () => {
      isMouseDown = false;
    };

    renderer.domElement.addEventListener('mousedown', onMouseDown);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('mouseup', onMouseUp);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      if (scene.children.length > 2) { // Has model loaded
        const model = scene.children[scene.children.length - 1];
        model.rotation.y += (targetRotationY - model.rotation.y) * 0.1;
        model.rotation.x += (targetRotationX - model.rotation.x) * 0.1;
      }
      
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (mountElement) {
        camera.aspect = mountElement.clientWidth / mountElement.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(mountElement.clientWidth, mountElement.clientHeight);
      }
    };
    window.addEventListener('resize', handleResize);

    return { scene, renderer, cleanup: () => {
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('mousedown', onMouseDown);
      renderer.domElement.removeEventListener('mousemove', onMouseMove);
      renderer.domElement.removeEventListener('mouseup', onMouseUp);
      renderer.dispose();
    }};
  };

  useEffect(() => {
    if (plyUrl && mountRef.current) {
      create3DViewer(mountRef.current, plyUrl).then(viewer => {
        sceneRef.current = viewer?.scene;
        rendererRef.current = viewer?.renderer;
      });
    }
    
    return () => {
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, [plyUrl]);

  useEffect(() => {
    if (showModelModal && plyUrl && modalMountRef.current) {
      create3DViewer(modalMountRef.current, plyUrl, true).then(viewer => {
        modalSceneRef.current = viewer?.scene;
        modalRendererRef.current = viewer?.renderer;
      });
    }
    
    return () => {
      if (modalRendererRef.current) {
        modalRendererRef.current.dispose();
      }
    };
  }, [showModelModal, plyUrl]);

  const LoaderComponent = () => (
    <div className="flex flex-col items-center justify-center space-y-4 py-12">
      <div className="w-12 h-12 border-4 border-gray-600 border-t-cyan-400 rounded-full animate-spin"></div>
      <p className="text-gray-300 text-lg">Generating 3D model...</p>
    </div>
  );

  const ModelModal = () => {
    if (!showModelModal || !plyUrl) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
        <div className="relative bg-gray-900 rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden border border-gray-700 shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <h3 className="text-xl font-semibold text-white">Generated 3D Model</h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleDownload}
                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                title="Download Model (.ply)"
              >
                <Download className="w-5 h-5" />
              </button>
              <button
                onClick={() => setShowModelModal(false)}
                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
          
          {/* 3D Model Viewer */}
          <div className="p-6 bg-gradient-to-br from-gray-800 to-gray-900">
            <div className="relative">
              <div 
                ref={modalMountRef}
                className="w-full h-96 rounded-xl border border-gray-600 bg-gray-900"
                style={{ minHeight: '400px' }}
              />
              <div className="absolute bottom-4 left-4 right-4 bg-black bg-opacity-60 backdrop-blur-sm rounded-lg p-3">
                <p className="text-white text-sm mb-2">
                  <span className="font-semibold">Prompt:</span> {prompt}
                </p>
                <p className="text-gray-300 text-xs">
                  <RotateCcw className="w-3 h-3 inline mr-1" />
                  Click and drag to rotate • Right-click and drag to pan
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <>
      <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-800 text-white flex flex-col items-center px-6 py-16">
        {/* Title */}
        <h1 className="text-5xl md:text-6xl font-bold mb-8 text-center bg-gradient-to-r from-white via-gray-200 to-pink-500 bg-clip-text text-transparent leading-tight">
          Generate Your 3D Model
        </h1>
        <p className="text-gray-300 text-lg md:text-xl max-w-3xl text-center mb-12">
          Type a prompt below and see your idea come to life as a 3D model that you can rotate and explore.
        </p>

        {/* Input + Button */}
        <div className="flex flex-col sm:flex-row gap-4 w-full max-w-3xl mb-12">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt... (e.g., 'A futuristic robot')"
            className="flex-1 px-6 py-4 rounded-full border border-gray-600 bg-gray-900/80 backdrop-blur-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent transition-all"
            onKeyPress={(e) => e.key === 'Enter' && handleGenerate()}
            disabled={loading}
          />
          <button
            onClick={handleGenerate}
            disabled={loading || !prompt}
            className="px-8 py-4 bg-gradient-to-r from-pink-700 to-pink-500 text-cyan-500 rounded-full font-semibold text-lg hover:scale-105 hover:shadow-xl hover:shadow-cyan-500/30 transition-all flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none"
          >
            <Play className="w-5 h-5" />
            {loading ? 'Generating...' : 'Generate'}
          </button>
        </div>

        {/* 3D Model Display Area - Always Visible */}
        <div className="w-full max-w-4xl">
          {/* Header */}
          <div className="text-center mb-6">
            <h2 className="text-2xl font-semibold mb-2 text-gray-200">
              {plyUrl && !loading ? "Your 3D Model is Ready!" : "Your 3D Model Will Appear Here"}
            </h2>
            {plyUrl && !loading && (
              <p className="text-gray-400">
                <RotateCcw className="w-4 h-4 inline mr-1" />
                Click and drag to rotate • Click the model to view in full screen
              </p>
            )}
          </div>

          {/* 3D Model Container */}
          <div className="relative min-h-[500px] rounded-2xl border-2 border-dashed border-gray-600 bg-gray-900/50 backdrop-blur-sm flex items-center justify-center overflow-hidden">
            {loading ? (
              <LoaderComponent />
            ) : plyUrl ? (
              <div 
                ref={mountRef}
                className="w-full h-full min-h-[500px] cursor-grab active:cursor-grabbing rounded-2xl"
                onClick={() => setShowModelModal(true)}
              />
            ) : (
              <div className="text-center text-gray-500">
                <div className="w-24 h-24 mx-auto mb-4 rounded-2xl border-2 border-gray-600 flex items-center justify-center">
                  <Play className="w-12 h-12 text-gray-600" />
                </div>
                <p className="text-lg mb-2">Enter a prompt and click Generate to create your 3D model</p>
                <p className="text-sm text-gray-600">Interactive 3D viewer with rotation controls</p>
              </div>
            )}
          </div>
          
          {/* Action buttons - Only show when model exists */}
          {plyUrl && !loading && (
            <div className="flex justify-center gap-4 mt-6">
              <button
                onClick={() => setShowModelModal(true)}
                className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
              >
                <Maximize2 className="w-4 h-4" />
                View Full Screen
              </button>
              <button
                onClick={handleDownload}
                className="px-6 py-3 bg-gradient-to-r from-pink-700 to-pink-500 hover:from-pink-700 hover:to-pink-500 text-white rounded-lg font-medium transition-all flex items-center gap-2 hover:shadow-lg"
              >
                <Download className="w-4 h-4" />
                Download PLY
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Model Modal */}
      <ModelModal />
    </>
  );
};

export default ImageGeneratorPage;