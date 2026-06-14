# E-Commerce Sales EDA — Weekend Project
## AI/ML Internship — ProSensia

### Dataset
10,000 e-commerce orders (2021–2024) across 26 columns including revenue, profit, shipping, customer segments, and product categories.

### Project Structure
| File | Description |
|------|-------------|
| `EDA_Ecommerce.ipynb` | Full EDA Jupyter Notebook |
| `ecommerce_cleaned.csv` | Cleaned dataset with parsed datetime |
| `requirements.txt` | Python dependencies |
| `chart1_overview.png` | Revenue, margin, region, trend dashboard |
| `chart2_heatmap.png` | Correlation heatmap |
| `chart3_distributions.png` | Distribution plots |
| `chart4_seasonal.png` | Seasonal & segment analysis |

### Key Findings
1. **Electronics = 64% of revenue** — dangerous concentration risk
2. **27.65% of orders returned or cancelled** — biggest profit leak
3. **Discount-profit correlation: -0.25** — discounts eroding margin without volume compensation
4. **All 4 regions within 6% revenue** — no dominant growth market
5. **VIP customers = 10% of orders but highest value per order** — underserved segment
6. **Shipping days range 3–25** — likely linked to high return rate

### How to Run
```bash
pip install -r requirements.txt
jupyter notebook EDA_Ecommerce.ipynb
```
