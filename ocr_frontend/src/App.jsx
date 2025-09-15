import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [recognizedText, setRecognizedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setRecognizedText('');
      setError('');
    }
  };

  const handleOcr = async () => {
    if (!selectedImage) {
      setError('Please select an image first.');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await axios.post('http://localhost:8000/api/ocr/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setRecognizedText(response.data.text);
    } catch (err) {
      setError('Error processing image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h1>Assamese OCR Web App</h1>
      <input type="file" onChange={handleImageChange} accept="image/*" />
      <button onClick={handleOcr} disabled={loading} style={{ margin: '10px' }}>
        {loading ? 'Processing...' : 'Recognize Text'}
      </button>

      {preview && (
        <div style={{ marginTop: '20px' }}>
          <h3>Image Preview:</h3>
          <img src={preview} alt="Preview" style={{ maxWidth: '400px', border: '1px solid #ccc' }} />
        </div>
      )}

      {error && <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>}

      {recognizedText && (
        <div style={{ marginTop: '20px', padding: '10px', border: '1px solid #ccc', borderRadius: '5px', backgroundColor: '#f0f0f0' }}>
          <h3>Recognized Text:</h3>
          <p style={{ whiteSpace: 'pre-wrap' }}>{recognizedText}</p>
        </div>
      )}
    </div>
  );
};

export default App;