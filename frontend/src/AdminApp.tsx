import { useState, useEffect } from 'react';
import AdminLogin from './AdminLogin';
import AdminSettings from './AdminSettings';
import { adminApi } from './adminApi';
import './Admin.css';

function AdminApp() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if we have a stored admin token
    const token = adminApi.getToken();
    if (token) {
      // Verify token is still valid by attempting to fetch config
      adminApi.getConfig()
        .then(() => {
          setIsAuthenticated(true);
        })
        .catch(() => {
          // Token expired or invalid
          adminApi.clearToken();
          setIsAuthenticated(false);
        })
        .finally(() => {
          setIsLoading(false);
        });
    } else {
      setIsLoading(false);
    }
  }, []);

  const handleLogin = (token: string) => {
    adminApi.setToken(token);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    adminApi.clearToken();
    setIsAuthenticated(false);
  };

  if (isLoading) {
    return (
      <div className="admin-loading">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <AdminLogin onLogin={handleLogin} />;
  }

  return <AdminSettings onLogout={handleLogout} />;
}

export default AdminApp;
