import { useState } from 'react';
import { api } from '../api';
import './Admin.css';

interface AdminProps {
  onLogout: () => void;
}

function Admin({ onLogout }: AdminProps) {
  const [activeTab, setActiveTab] = useState<'users' | 'keys'>('users');

  return (
    <div className="admin-container">
      <header className="admin-header">
        <h1>Admin Panel</h1>
        <button onClick={onLogout} className="logout-button">
          Logout
        </button>
      </header>

      <div className="admin-tabs">
        <button
          className={activeTab === 'users' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('users')}
        >
          Users
        </button>
        <button
          className={activeTab === 'keys' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('keys')}
        >
          API Keys
        </button>
      </div>

      <div className="admin-content">
        {activeTab === 'users' ? (
          <div className="panel">
            <h2>User Management</h2>
            <p>Create and manage users here</p>
          </div>
        ) : (
          <div className="panel">
            <h2>API Key Management</h2>
            <p>Generate and revoke API keys here</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Admin;
