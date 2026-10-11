import type { Metadata } from "next";
import "./globals.css";
import Providers from "@/components/Providers";

export const metadata: Metadata = {
  title: "EdgeMind — Industrial Edge Vector Intelligence",
  description: "Enterprise flat-style industrial offline vector memory and air-gapped CRDT peer mesh console",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#F8FAFC] text-[#0F172A] min-h-screen relative antialiased selection:bg-blue-100 selection:text-blue-900">
        <div className="relative z-10 min-h-screen">
          <Providers>{children}</Providers>
        </div>
      </body>
    </html>
  );
}
