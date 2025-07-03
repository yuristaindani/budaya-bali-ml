import type { Metadata } from "next";
import { Montserrat } from "next/font/google";
import "./globals.css";

const montserrat = Montserrat({ subsets: ["latin"], weight: ["400", "700"] });

export const metadata: Metadata = {
  title: "Budaya Bali",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="id">
      <body className={montserrat.className + " antialiased bg-[#f4f4f4]"}>
        {children}
      </body>
    </html>
  );
}