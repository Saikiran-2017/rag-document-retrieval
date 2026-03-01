import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Smaller production image when using web/Dockerfile
  output: "standalone",
};

export default nextConfig;
