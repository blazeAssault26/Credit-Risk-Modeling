Credit Risk Modeling Project
Understanding the Problem: What Is Credit Risk?
Credit risk refers to the likelihood that a borrower will fail to repay their loan on time. In banking terms, we track this risk through DPD (Days Past Due)—the number of days a loan payment is overdue. If the DPD exceeds a specific threshold (e.g., 90 days), it can be classified as an NPA (Non-Performing Asset), indicating a loan that has defaulted and is unlikely to be repaid. For the bank, NPAs represent a financial loss.

Portfolio at Risk (PAR) is a key metric that banks use to measure the outstanding principal (OSP) for loans that are past due. Managing this risk is crucial for the health of a bank’s loan portfolio.

The Data We Used: Combining Internal and External Sources
Our model was trained on two datasets:

CIBIL Data (60 Features)
This dataset consolidates a person’s credit history across multiple banks and includes:

Credit Score: Ranges from 350 to 900, reflecting the borrower’s creditworthiness, where higher scores indicate lower risk. Our dataset’s scores range from 469 to 811.
Trade Lines (TL): Contains details on all loans and credit accounts held by the individual across various banks, providing insights into total debt exposure and repayment behaviors.
Payment History: Records timely or missed payments over various periods (e.g., 6 and 12 months), crucial for assessing the likelihood of default. This also tracks key delinquency indicators, such as Days Past Due (DPD), which help measure short-term risk.
Internal Bank Data (25 Features)
This dataset provides a more granular view of the applicant's relationship with our bank:

Total Trade Lines and Active Trade Lines: Metrics that reflect the customer’s engagement with the bank’s credit products, indicating their level of leverage.
Assets and Liabilities: Includes data on assets like savings and deposits, offering a comprehensive view of the customer’s financial standing and assisting in assessing their capacity to take on additional debt.
