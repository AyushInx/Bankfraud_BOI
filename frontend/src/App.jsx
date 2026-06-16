// src/App.jsx — Root app with router

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Upload from './pages/Upload';
import Investigate from './pages/Investigate';
import Alerts from './pages/Alerts';
import './index.css';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/"            element={<Home />}        />
            <Route path="/upload"      element={<Upload />}      />
            <Route path="/investigate" element={<Investigate />} />
            <Route path="/alerts"      element={<Alerts />}      />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
