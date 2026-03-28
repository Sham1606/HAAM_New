
import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Activity, LogOut, Shield, User } from 'lucide-react';
import { useAuth } from '../../services/AuthContext';

const Navbar = () => {
    const [isOpen, setIsOpen] = useState(false);
    const location = useLocation();
    const { user, logout, isAdmin } = useAuth();

    // ── Role-based navigation ────────────────────────────────────────────────
    const adminLinks = [
        { name: 'Calls', path: '/' },
        { name: 'Agents', path: '/agents' },
        { name: 'Analytics', path: '/analytics' },
        { name: 'Agent Grid', path: '/agent-grid' },
        { name: 'Alerts', path: '/alerts' },
    ];

    const agentLinks = [
        { name: 'My Dashboard', path: '/' },
        { name: 'Live Analysis', path: '/live' },
        { name: 'My Calls', path: '/my-calls' },
        { name: 'My Feedback', path: '/my-feedback' },
    ];

    const navLinks = isAdmin ? adminLinks : agentLinks;

    const isActive = (path) => {
        return location.pathname === path ? 'text-primary border-b-2 border-primary' : 'text-gray-600 hover:text-primary';
    };

    return (
        <nav className="bg-white shadow-sm">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    <div className="flex">
                        <div className="flex-shrink-0 flex items-center">
                            <Activity className="h-8 w-8 text-primary" />
                            <span className="ml-2 text-xl font-bold text-gray-800">HAAM Framework</span>
                        </div>
                        <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                            {navLinks.map((link) => (
                                <Link
                                    key={link.name}
                                    to={link.path}
                                    className={`inline-flex items-center px-1 pt-1 text-sm font-medium ${isActive(link.path)}`}
                                >
                                    {link.name}
                                </Link>
                            ))}
                        </div>
                    </div>

                    {/* User Info + Logout */}
                    <div className="hidden sm:flex items-center gap-3">
                        {user && (
                            <>
                                <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 rounded-lg border border-gray-100">
                                    {isAdmin ? (
                                        <Shield className="h-3.5 w-3.5 text-purple-500" />
                                    ) : (
                                        <User className="h-3.5 w-3.5 text-indigo-500" />
                                    )}
                                    <span className="text-xs font-semibold text-gray-700">
                                        {user.display_name || user.username}
                                    </span>
                                    <span className={`text-xs px-1.5 py-0.5 rounded-full font-medium ${isAdmin ? 'bg-purple-100 text-purple-600' : 'bg-blue-100 text-blue-600'
                                        }`}>
                                        {user.role}
                                    </span>
                                </div>
                                <button
                                    onClick={logout}
                                    className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition"
                                    title="Sign out"
                                >
                                    <LogOut className="h-3.5 w-3.5" />
                                    Logout
                                </button>
                            </>
                        )}
                    </div>

                    <div className="-mr-2 flex items-center sm:hidden">
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary"
                        >
                            {isOpen ? <X className="block h-6 w-6" /> : <Menu className="block h-6 w-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile menu */}
            {isOpen && (
                <div className="sm:hidden">
                    <div className="pt-2 pb-3 space-y-1">
                        {navLinks.map((link) => (
                            <Link
                                key={link.name}
                                to={link.path}
                                className={`block pl-3 pr-4 py-2 border-l-4 text-base font-medium ${location.pathname === link.path
                                    ? 'bg-blue-50 border-primary text-primary'
                                    : 'border-transparent text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800'
                                    }`}
                                onClick={() => setIsOpen(false)}
                            >
                                {link.name}
                            </Link>
                        ))}
                        {user && (
                            <button
                                onClick={() => { logout(); setIsOpen(false); }}
                                className="block w-full text-left pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-red-600 hover:bg-red-50"
                            >
                                <LogOut className="h-4 w-4 inline mr-2" />
                                Logout ({user.display_name || user.username})
                            </button>
                        )}
                    </div>
                </div>
            )}
        </nav>
    );
};

export default Navbar;
