import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MineLens AI — Critical Mineral Prospectivity Mapping",
  description:
    "Interactive AI-powered geoscience dashboard for critical mineral exploration. Powered by Gemma 4's function calling capabilities.",
  keywords: [
    "MineLens",
    "Gemma 4",
    "mineral exploration",
    "geoscience",
    "critical minerals",
    "lithium",
    "cobalt",
    "rare earth",
    "copper",
    "nickel",
  ],
  authors: [{ name: "MineLens AI Team" }],
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔬</text></svg>",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
        style={{ background: "#0a0f1a", color: "#e2e8f0" }}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
