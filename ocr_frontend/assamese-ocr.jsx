import React, { useState } from 'react';

const AssameseOCR = () => {
    const [image, setImage] = useState(null);
    const [result, setResult] = useState('');
    const [error, setError] = useState('');

    const handleImageChange = (e) => {
        setImage(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!image) {
            setError('Please upload an image.');
            return;
        }

        const formData = new FormData();
        formData.append('image', image);

        try {
            const response = await fetch('http://localhost:8000/api/ocr/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong');
            }

            const data = await response.json();
            setResult(data.text);
            setError('');
        } catch (err) {
            setError(err.message);
            setResult('');
        }
    };

    return (
        <div className="ocr-container">
            <form onSubmit={handleSubmit}>
                <input type="file" accept="image/*" onChange={handleImageChange} />
                <button type="submit">Upload and Process</button>
            </form>
            {error && <p className="error">{error}</p>}
            {result && <p className="result">{result}</p>}
        </div>
    );
};

export default AssameseOCR;
