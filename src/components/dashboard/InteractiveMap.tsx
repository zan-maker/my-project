"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapPin, Crosshair } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Location } from "@/lib/types";

interface InteractiveMapProps {
  selectedLocation: Location | null;
  onLocationSelect: (lat: number, lon: number) => void;
  className?: string;
}

// Custom marker icon
const customIcon = L.divIcon({
  className: "custom-marker",
  html: `
    <div style="
      width: 32px; height: 32px; position: relative;
      display: flex; align-items: center; justify-content: center;
    ">
      <div style="
        width: 32px; height: 32px; border-radius: 50%;
        background: rgba(16,185,129,0.3); border: 2px solid #10b981;
        animation: pulse-ring 2s ease-out infinite;
        position: absolute;
      "></div>
      <div style="
        width: 14px; height: 14px; border-radius: 50%;
        background: #10b981; border: 2px solid #0a0f1a;
        z-index: 2; box-shadow: 0 0 12px rgba(16,185,129,0.6);
      "></div>
    </div>
    <style>
      @keyframes pulse-ring {
        0% { transform: scale(1); opacity: 1; }
        100% { transform: scale(2.5); opacity: 0; }
      }
    </style>
  `,
  iconSize: [32, 32],
  iconAnchor: [16, 16],
});

// Mineral deposit markers
const depositColors: Record<string, string> = {
  lithium: "#10b981",
  cobalt: "#6366f1",
  rare_earth: "#f59e0b",
  copper: "#ef4444",
  nickel: "#8b5cf6",
};

export default function InteractiveMap({
  selectedLocation,
  onLocationSelect,
  className = "",
}: InteractiveMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const onLocationSelectRef = useRef(onLocationSelect);

  useEffect(() => {
    onLocationSelectRef.current = onLocationSelect;
  }, [onLocationSelect]);

  // Initialize map
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = L.map(mapRef.current, {
      center: [-25.0, 135.0], // Center of Australia
      zoom: 4,
      zoomControl: false,
      attributionControl: false,
    });

    // Dark-themed tile layer
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19,
    }).addTo(map);

    // Add some sample mineral deposit markers
    const deposits = [
      { lat: -32.85, lon: 116.15, mineral: "lithium", name: "Greenbushes" },
      { lat: -24.16, lon: 120.23, mineral: "nickel", name: "Mount Keith" },
      { lat: -31.21, lon: 121.81, mineral: "gold", name: "Kalgoorlie" },
      { lat: -20.72, lon: 117.58, mineral: "rare_earth", name: "Weld Range" },
      { lat: -29.01, lon: 167.95, mineral: "nickel", name: "Norilsk" },
      { lat: -22.29, lon: 137.95, mineral: "copper", name: "Olympic Dam" },
      { lat: -11.65, lon: -68.75, mineral: "cobalt", name: "Bolivia" },
      { lat: 36.63, lon: -108.12, mineral: "rare_earth", name: "Mountain Pass" },
      { lat: -24.7, lon: 70.15, mineral: "copper", name: "Chuquicamata" },
      { lat: -12.05, lon: 28.3, mineral: "cobalt", name: "Kamoto" },
      { lat: 40.73, lon: -107.6, mineral: "lithium", name: "Thacker Pass" },
      { lat: 46.28, lon: 88.13, mineral: "rare_earth", name: "Bayan Obo" },
    ];

    deposits.forEach((d) => {
      const color = depositColors[d.mineral] || "#10b981";
      L.circleMarker([d.lat, d.lon], {
        radius: 5,
        fillColor: color,
        color: color,
        weight: 1,
        opacity: 0.7,
        fillOpacity: 0.5,
      })
        .addTo(map)
        .bindTooltip(d.name, {
          className: "dark-tooltip",
          direction: "top",
          offset: [0, -8],
        });
    });

    // Click handler
    map.on("click", (e: L.LeafletMouseEvent) => {
      onLocationSelectRef.current(e.latlng.lat, e.latlng.lng);
    });

    mapInstanceRef.current = map;

    // Add zoom control to bottom right
    L.control
      .zoom({ position: "bottomright" })
      .addTo(map);

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update marker when location changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (markerRef.current) {
      map.removeLayer(markerRef.current);
      markerRef.current = null;
    }

    if (selectedLocation) {
      markerRef.current = L.marker(
        [selectedLocation.lat, selectedLocation.lon],
        { icon: customIcon }
      )
        .addTo(map)
        .bindTooltip(
          `${selectedLocation.lat.toFixed(4)}°, ${selectedLocation.lon.toFixed(4)}°`,
          {
            className: "dark-tooltip",
            direction: "top",
            offset: [0, -20],
            permanent: true,
          }
        );
    }
  }, [selectedLocation]);

  return (
    <div className={`relative ${className}`}>
      <div ref={mapRef} className="w-full h-full rounded-xl" />

      {/* Map overlay controls */}
      <div className="absolute top-3 left-3 z-[1000] flex flex-col gap-2">
        <div className="bg-[#0a0f1a]/90 backdrop-blur-md rounded-lg px-3 py-2 border border-white/10">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <MapPin className="w-3.5 h-3.5 text-emerald-400" />
            <span>
              {selectedLocation
                ? `${selectedLocation.lat.toFixed(4)}°, ${selectedLocation.lon.toFixed(4)}°`
                : "Click to select location"}
            </span>
          </div>
        </div>
      </div>

      <div className="absolute top-3 right-3 z-[1000] flex gap-2">
        <Button
          size="sm"
          variant="ghost"
          className="bg-[#0a0f1a]/90 backdrop-blur-md border border-white/10 text-slate-300 hover:text-emerald-400 hover:bg-[#0a0f1a]/95 h-8 w-8 p-0"
          onClick={() => {
            navigator.geolocation?.getCurrentPosition(
              (pos) => {
                const { latitude, longitude } = pos.coords;
                onLocationSelect(latitude, longitude);
                mapInstanceRef.current?.flyTo([latitude, longitude], 10, {
                  duration: 1.5,
                });
              },
              () => {}
            );
          }}
        >
          <Crosshair className="w-4 h-4" />
        </Button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-3 left-3 z-[1000]">
        <div className="bg-[#0a0f1a]/90 backdrop-blur-md rounded-lg px-3 py-2 border border-white/10">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">
            Known Deposits
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(depositColors).map(([mineral, color]) => (
              <div key={mineral} className="flex items-center gap-1">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ background: color, boxShadow: `0 0 6px ${color}80` }}
                />
                <span className="text-[10px] text-slate-400 capitalize">
                  {mineral.replace("_", " ")}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <style jsx global>{`
        .dark-tooltip {
          background: #0a0f1a !important;
          border: 1px solid rgba(255,255,255,0.15) !important;
          color: #94a3b8 !important;
          font-size: 11px !important;
          padding: 4px 8px !important;
          border-radius: 6px !important;
          box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
        }
        .dark-tooltip::before {
          border-top-color: rgba(255,255,255,0.15) !important;
        }
        .leaflet-control-zoom a {
          background: #0a0f1a !important;
          color: #94a3b8 !important;
          border-color: rgba(255,255,255,0.1) !important;
        }
        .leaflet-control-zoom a:hover {
          background: #1a2332 !important;
          color: #10b981 !important;
        }
        .leaflet-container {
          background: #0a0f1a !important;
        }
      `}</style>
    </div>
  );
}
