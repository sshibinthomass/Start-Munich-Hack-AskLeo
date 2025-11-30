import React, { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import "./ProductLanding.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export function ProductLanding() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedReviews, setSelectedReviews] = useState(null);
  const [selectedProductName, setSelectedProductName] = useState("");
  const [selectedPriceHistory, setSelectedPriceHistory] = useState(null);
  const [selectedPriceHistoryProductName, setSelectedPriceHistoryProductName] = useState("");

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        // Try to fetch from backend API first with cache-busting timestamp
        const timestamp = new Date().getTime();
        const response = await fetch(`${BACKEND_URL}/products?t=${timestamp}`, {
          cache: 'no-cache',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        });
        if (response.ok) {
          const data = await response.json();
          setProducts(data);
        } else {
          // Fallback: try to load from static file
          const staticResponse = await fetch("/data/product.json", {
            cache: 'no-cache',
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          });
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
            brand: "BrewBot",
            color: "Black",
            currency: "USD",
            versions: [
              {
                version: "Gravimetric 3gr",
                price: 32900.0,
                minPrice: 29610.0,
                maxPrice: 32900.0,
                description: "BrewBot's top flagship machine featuring gravimetric technology, ultra-precise extraction control, and modern angular design.",
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
            brand: "BrewBot",
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
            brand: "BrewBot",
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

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", { month: "short", year: "numeric" });
  };

  const preparePriceHistoryData = (priceHistory) => {
    return priceHistory.map(item => ({
      ...item,
      formattedDate: formatDate(item.date)
    }));
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
          <div className="product-landing__logo-container">
            <img 
              src={`${BACKEND_URL}/api/images/icon.png`} 
              alt="BrewBot Logo" 
              className="product-landing__logo"
              onError={(e) => {
                e.target.style.display = 'none';
              }}
            />
            <h1 className="product-landing__title">BrewBot</h1>
          </div>
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
                  
                  <h3 className="product-card__name">
                    {product.name}
                  </h3>
                  <div className="product-card__brand-price">
                    <p className="product-card__brand">{product.brand}</p>
                    <span 
                      className="product-card__price product-card__price--clickable"
                      onClick={async () => {
                        // Fetch fresh data when opening price history
                        try {
                          const timestamp = new Date().getTime();
                          const response = await fetch(`${BACKEND_URL}/products?t=${timestamp}`, {
                            cache: 'no-cache',
                            headers: {
                              'Cache-Control': 'no-cache',
                              'Pragma': 'no-cache'
                            }
                          });
                          if (response.ok) {
                            const freshData = await response.json();
                            const freshProduct = freshData.find(p => p.id === product.id);
                            const freshVersion = freshProduct?.versions?.[0];
                            if (freshVersion?.priceHistory && freshVersion.priceHistory.length > 0) {
                              setSelectedPriceHistory(freshVersion.priceHistory);
                              setSelectedPriceHistoryProductName(product.name);
                            } else if (version.priceHistory && version.priceHistory.length > 0) {
                              // Fallback to current state if fresh fetch fails
                              setSelectedPriceHistory(version.priceHistory);
                              setSelectedPriceHistoryProductName(product.name);
                            }
                          } else {
                            // Fallback to current state
                            if (version.priceHistory && version.priceHistory.length > 0) {
                              setSelectedPriceHistory(version.priceHistory);
                              setSelectedPriceHistoryProductName(product.name);
                            }
                          }
                        } catch (err) {
                          // Fallback to current state on error
                          if (version.priceHistory && version.priceHistory.length > 0) {
                            setSelectedPriceHistory(version.priceHistory);
                            setSelectedPriceHistoryProductName(product.name);
                          }
                        }
                      }}
                    >
                      {formatPrice(version.price)}
                    </span>
                  </div>
                  
                  <div 
                    className="product-card__rating product-card__rating--clickable"
                    onClick={() => {
                      if (version.reviewsList && version.reviewsList.length > 0) {
                        setSelectedReviews(version.reviewsList);
                        setSelectedProductName(product.name);
                      }
                    }}
                  >
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

      {/* Reviews Modal */}
      {selectedReviews && (
        <div className="reviews-modal-overlay" onClick={() => setSelectedReviews(null)}>
          <div className="reviews-modal" onClick={(e) => e.stopPropagation()}>
            <div className="reviews-modal__header">
              <h2 className="reviews-modal__title">Customer Reviews - {selectedProductName}</h2>
              <button 
                className="reviews-modal__close"
                onClick={() => setSelectedReviews(null)}
                aria-label="Close reviews"
              >
                ×
              </button>
            </div>
            <div className="reviews-modal__content">
              {selectedReviews.map((review, index) => (
                <div key={index} className="review-item">
                  <div className="review-item__header">
                    <div className="review-item__author">{review.author}</div>
                    <div className="review-item__rating">
                      {"★".repeat(review.rating)}{"☆".repeat(5 - review.rating)}
                    </div>
                    <div className="review-item__date">{review.date}</div>
                  </div>
                  <div className="review-item__comment">{review.comment}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Price History Modal */}
      {selectedPriceHistory && (
        <div className="price-history-modal-overlay" onClick={() => setSelectedPriceHistory(null)}>
          <div className="price-history-modal" onClick={(e) => e.stopPropagation()}>
            <div className="price-history-modal__header">
              <h2 className="price-history-modal__title">Price History - {selectedPriceHistoryProductName}</h2>
              <button 
                className="price-history-modal__close"
                onClick={() => setSelectedPriceHistory(null)}
                aria-label="Close price history"
              >
                ×
              </button>
            </div>
            <div className="price-history-modal__content">
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={preparePriceHistoryData(selectedPriceHistory)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="formattedDate" 
                    tick={{ fontSize: 12 }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => formatPrice(value)}
                  />
                  <Tooltip 
                    formatter={(value) => formatPrice(value)}
                    labelStyle={{ color: '#1f2937' }}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e5e7eb' }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#0066cc" 
                    strokeWidth={3}
                    dot={{ fill: '#0066cc', r: 5 }}
                    activeDot={{ r: 8 }}
                    name="Price"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

