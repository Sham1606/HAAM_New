
import React from 'react';

const CallFilters = ({ onFilterChange }) => {
    const handleChange = (key, value) => {
        onFilterChange(prev => ({ ...prev, [key]: value }));
    };

    return (
        <div className="bg-white p-4 rounded-lg shadow-sm flex flex-wrap gap-4 items-center">
            <input
                type="text"
                placeholder="Filter by Agent ID"
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                onChange={(e) => handleChange('agent_id', e.target.value)}
            />

            {/* <input
        type="date"
        className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
        onChange={(e) => handleChange('date_from', e.target.value)}
      /> */}

            <select
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                onChange={(e) => handleChange('emotion', e.target.value)}
            >
                <option value="">All Emotions</option>
                <option value="joy">Joy</option>
                <option value="anger">Anger</option>
                <option value="sadness">Sadness</option>
                <option value="neutral">Neutral</option>
            </select>
        </div>
    );
};

export default CallFilters;
