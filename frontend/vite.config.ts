import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In dev the FastAPI backend runs on :8000; proxy /api there so the frontend
// can call relative URLs (CORS is a non-issue locally, and prod can point
// VITE at a real host).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
