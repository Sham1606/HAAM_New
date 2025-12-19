
import React, { useState, useEffect } from 'react';
import { Search, X } from 'lucide-react';

const SearchBar = ({ onSearch, placeholder = "Search..." }) => {
    const [term, setTerm] = useState('');

    useEffect(() => {
        const timer = setTimeout(() => {
            onSearch(term);
        }, 500);

        return () => clearTimeout(timer);
    }, [term, onSearch]);

    return (
        <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
            </div>
            <input
                type="text"
                className="block w-full pl-10 pr-10 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-primary focus:border-primary sm:text-sm"
                placeholder={placeholder}
                value={term}
                onChange={(e) => setTerm(e.target.value)}
            />
            {term && (
                <button
                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    onClick={() => setTerm('')}
                >
                    <X className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                </button>
            )}
        </div>
    );
};

export default SearchBar;
