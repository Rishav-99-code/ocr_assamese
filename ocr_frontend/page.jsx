import React from 'react';
import AssameseOCR from './assamese-ocr';
import './global.css';

const Page = () => {
    return (
        <div className="app-container">
            <h1>Assamese OCR</h1>
            <AssameseOCR />
        </div>
    );
};

export default Page;
