# Credit-risk-model
Credit Risk Probability Model for Alternative Data
## Task 1: Credit Scoring Business Understanding

### 1. Basel II Accord and Model Requirements

The Basel II Capital Accord fundamentally shapes how banks approach credit risk modeling through its three-pillar framework:

**Pillar 1 - Minimum Capital Requirements (Direct Impact on Our Model):**
- **Risk-Weighted Assets (RWA)** calculation: RWA = EAD × PD × LGD
- Our model must provide **Probability of Default (PD)** estimates for each customer
- **Accuracy Requirement**: Capital reserves are directly proportional to PD estimates
- **Interpretability Need**: Regulators must be able to validate how PD estimates are derived

**Pillar 2 - Supervisory Review (Model Governance):**
- Regulators review **Internal Ratings-Based (IRB)** approaches
- Our model must have:
  - Clear documentation of development process
  - Comprehensive validation framework
  - Ongoing monitoring procedures
  - Back-testing against actual outcomes

**Pillar 3 - Market Discipline (Transparency):**
- Requires disclosure of risk assessment methodologies
- Stakeholders need to understand how credit decisions are made
- Builds trust in the BNPL service

**Specific Implications for This Project:**
1. **Document Every Assumption**: Why RFM metrics are suitable proxies for credit risk
2. **Model Transparency**: Clear explanation of feature importance and decision logic
3. **Validation Framework**: How we'll validate proxy-based predictions
4. **Compliance Records**: Maintain audit trail of all model decisions

### 2. Proxy Variable Necessity and Business Risks for Bati Bank's BNPL Service

**Why Proxy is Necessary in This Context:**
- **Data Reality**: E-commerce platform provides transaction data but no repayment history
- **Innovation Opportunity**: Alternative data can assess thin-file/no-file customers
- **Behavioral Economics**: Research indicates spending habits correlate with financial responsibility
- **RFM as Credit Proxy**: Recency (engagement), Frequency (consistency), Monetary (spending capacity)

**Business Risks and Mitigation Strategies:**

| Risk | Impact on Bati Bank | Mitigation Strategy |
|------|---------------------|---------------------|
| **False Positives (Type I)** | Reject creditworthy customers → Lost revenue | Start with conservative thresholds, gradual relaxation |
| **False Negatives (Type II)** | Approve risky customers → Default losses | Implement credit limits, phased rollout |
| **Proxy Drift** | Changing e-commerce behavior ≠ changing credit risk | Monthly model recalibration, monitoring feature stability |
| **Regulatory Challenge** | Regulators skeptical of alternative data | Comprehensive documentation, academic references |
| **Data Bias** | Platform users ≠ general population | Demographic analysis, fairness testing |
| **Model Overfit** | Works on historical data but fails on new customers | Robust cross-validation, out-of-time testing |

**Implementation Strategy:**
1. **Pilot Phase**: Start with small credit limits to test proxy effectiveness
2. **Performance Tracking**: Monitor actual repayment rates vs. predicted risk
3. **Feedback Loop**: Update model as real repayment data accumulates
4. **Human-in-the-Loop**: Manual review for edge cases

### 3. Model Selection Trade-offs for BNPL Service

**BNPL-Specific Considerations:**
- **High Volume, Low Value**: Many small transactions vs. few large loans
- **Speed Requirement**: Real-time credit decisions at checkout
- **Customer Experience**: Quick approval process essential
- **Regulatory Scrutiny**: New BNPL regulations emerging globally

**Model Comparison for BNPL Context:**

**Logistic Regression with WoE:**
✅ Advantages for BNPL:

- Regulatory-friendly: Clear audit trail
- Fast inference: ~1ms per prediction
- Stable: Consistent behavior over time
- Explainable: "Customer declined due to low frequency score"

⚠️ Limitations:

- May miss complex spending pattern interactions
- Requires careful feature engineering

**Gradient Boosting (XGBoost/LightGBM):**
✅ Advantages for BNPL:

- High accuracy: Captures complex behavioral patterns
- Handles feature interactions automatically
- Robust to outliers in transaction data

⚠️ Challenges:

- Black box: Hard to explain to customers
- Regulatory approval may take longer
- Higher computational cost for real-time scoring

**Recommended Hybrid Approach:**
1. **Initial Deployment**: Logistic Regression with WoE
   - Faster regulatory approval
   - Establish baseline performance
   - Build customer trust through transparency

2. **Performance Enhancement**: Gradient Boosting ensemble
   - Use as secondary validation
   - Identify edge cases
   - Gradually increase weight based on performance

3. **Explainability Bridge**: Implement SHAP values
   - Provide feature importance for complex models
   - Maintain regulatory compliance
   - Customer-facing explanations

**Key Decision Factors for Bati Bank:**
1. **Regulatory Timeline**: If quick deployment is critical → Logistic Regression
2. **Performance Priority**: If maximizing approval accuracy is key → Gradient Boosting + explainability
3. **Resource Availability**: Computational infrastructure for real-time scoring
4. **Risk Appetite**: Conservative vs. aggressive growth strategy

**Conclusion**: Given BNPL's emerging regulatory landscape and need for customer trust, starting with an interpretable Logistic Regression model while developing Gradient Boosting as a challenger model provides the optimal balance between compliance and performance.