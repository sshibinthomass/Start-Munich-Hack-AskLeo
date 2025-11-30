import React, { useState, useEffect } from "react";
import "./ProductLanding.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export function ProductLanding() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        // Try to fetch from backend API first
        const response = await fetch(`${BACKEND_URL}/products`);
        if (response.ok) {
          const data = await response.json();
          setProducts(data);
        } else {
          // Fallback: try to load from static file
          const staticResponse = await fetch("/data/product.json");
          if (staticResponse.ok) {
            const data = await staticResponse.json();
            setProducts(data);
          } else {
            throw new Error("Failed to load products");
          }
        }
      } catch (err) {
        // If all fails, use hardcoded data
        console.warn("Could not fetch products, using fallback data:", err);
        setProducts([
          {
            id: "va_prod_001",
            name: "Maverick",
            category: "Flagship Espresso Machine",
            brand: "Victoria Arduino",
            color: "Black",
            currency: "USD",
            versions: [
              {
                version: "Gravimetric 3gr",
                price: 32900.0,
                minPrice: 29610.0,
                maxPrice: 32900.0,
                description: "Victoria Arduino's top flagship machine featuring gravimetric technology, ultra-precise extraction control, and modern angular design.",
                inStock: true,
                stockQuantity: 5,
                rating: 4.9,
                reviews: 42,
                features: [
                  "Gravimetric extraction technology",
                  "Multi-boiler T3 system",
                  "Independent temperature profiling",
                ],
              },
            ],
          },
          {
            id: "va_prod_002",
            name: "Eagle Tempo",
            category: "Mid-Range Espresso Machine",
            brand: "Victoria Arduino",
            color: "Black",
            currency: "USD",
            versions: [
              {
                version: "Digit 3gr",
                price: 19900.0,
                minPrice: 17900.0,
                maxPrice: 19900.0,
                description: "A modern, stylish, and high-efficiency espresso machine designed for fast-paced cafés.",
                inStock: true,
                stockQuantity: 12,
                rating: 4.7,
                reviews: 67,
                features: [
                  "Digit electronic dosing",
                  "Modern minimalist design",
                  "High-efficiency thermal system",
                ],
              },
            ],
          },
          {
            id: "va_prod_003",
            name: "E1 Prima",
            category: "Entry-Level Compact Espresso Machine",
            brand: "Victoria Arduino",
            color: "White",
            currency: "USD",
            versions: [
              {
                version: "Volumetric T3 w/ Easy Cream",
                price: 7490.0,
                minPrice: 6990.0,
                maxPrice: 7490.0,
                description: "A compact, beautifully designed single-group espresso machine featuring full T3 temperature control.",
                inStock: true,
                stockQuantity: 18,
                rating: 4.6,
                reviews: 94,
                features: [
                  "T3 multi-temperature control",
                  "Easy Cream automatic milk steaming",
                  "Compact footprint",
                ],
              },
            ],
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchProducts();
  }, []);

  const formatPrice = (price) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price);
  };

  if (loading) {
    return (
      <div className="product-landing">
        <div className="product-landing__loading">Loading products...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="product-landing">
        <div className="product-landing__error">{error}</div>
      </div>
    );
  }

  return (
    <div className="product-landing">
      <header className="product-landing__header">
        <div className="product-landing__header-content">
          <h1 className="product-landing__title">BrewBot</h1>
          <p className="product-landing__subtitle">AI-Powered Procurement Solutions</p>
        </div>
      </header>

      <main className="product-landing__main">
        <div className="product-landing__container">
          <h2 className="product-landing__section-title">Featured Products</h2>
          <div className="product-grid">
            {products.map((product) => {
              const version = product.versions?.[0];
              if (!version) return null;

              return (
                <div key={product.id} className="product-card">
                  <div className="product-card__header">
                    <span className="product-card__category">{product.category}</span>
                  </div>
                  
                  {version.images && version.images.length > 0 && (
                    <div className="product-card__image-container">
                      <img 
                        src={`${BACKEND_URL}${version.images[0]}`} 
                        alt={product.name}
                        className="product-card__image"
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                    </div>
                  )}
                  
                  <h3 className="product-card__name">{product.name}</h3>
                  <p className="product-card__brand">{product.brand}</p>
                  
                  <div className="product-card__rating">
                    <span className="product-card__stars">{"★".repeat(Math.floor(version.rating))}</span>
                    <span className="product-card__rating-text">
                      {version.rating} ({version.reviews} reviews)
                    </span>
                  </div>

                  <p className="product-card__description">{version.description}</p>

                  <div className="product-card__footer">
                    <button className="product-card__button">Learn More</button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </main>
    </div>
  );
}

