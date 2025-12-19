
import React, { useEffect, useState } from 'react';

const ErrorToast = ({ message, onClose }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onClose();
        }, 3000);
        return () => clearTimeout(timer);
    }, [onClose]);

    return (
        <div className="fixed top-4 right-4 bg-danger text-white px-6 py-3 rounded shadow-lg z-50 animate-bounce">
            {message}
        </div>
    );
};

export default ErrorToast;
