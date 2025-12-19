
import React from 'react';

const RiskBadge = ({ level }) => {
    const styles = {
        low: 'bg-green-500 text-white',
        medium: 'bg-yellow-500 text-white',
        high: 'bg-orange-500 text-white',
        critical: 'bg-red-500 text-white animate-pulse',
    };

    const style = styles[level?.toLowerCase()] || 'bg-gray-400 text-white';

    return (
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${style} capitalize shadow-sm`}>
            {level}
        </span>
    );
};

export default RiskBadge;
