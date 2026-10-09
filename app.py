import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Risk System",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("employee_attrition_risk.csv")


@st.cache_resource
def load_model():
    return joblib.load("employee_attrition_model.pkl")


df = load_data()
model = load_model()

# ============================================================
# TITLE
# ============================================================

st.title("📊 Employee Attrition Risk Scoring System")

st.write(
    "Machine Learning–Based Employee Attrition Prediction "
    "and Risk Scoring Dashboard"
)

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

page = st.sidebar.radio(
    "Select a page:",
    [
        "Overview Dashboard",
        "Employee Risk Profile",
        "Department Risk Analysis",
        "Explainability",
        "What-If Analysis"
    ]
)

# ============================================================
# FILTERS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")

# Department filter
departments = ["All"] + sorted(
    df["Department"].dropna().unique().tolist()
)

selected_department = st.sidebar.selectbox(
    "Department",
    departments
)

# Job role filter
roles = ["All"] + sorted(
    df["JobRole"].dropna().unique().tolist()
)

selected_role = st.sidebar.selectbox(
    "Job Role",
    roles
)

# Minimum risk threshold
risk_threshold = st.sidebar.slider(
    "Minimum Risk Score (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=5
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[
        filtered_df["Department"] == selected_department
    ]

if selected_role != "All":
    filtered_df = filtered_df[
        filtered_df["JobRole"] == selected_role
    ]

filtered_df = filtered_df[
    filtered_df["RiskScore"] >= risk_threshold
]

# ============================================================
# OVERVIEW DASHBOARD
# ============================================================

if page == "Overview Dashboard":

    st.header("🏠 Attrition Risk Overview")
    st.markdown("---")

    # ========================================================
    # PROJECT SUMMARY
    # ========================================================

    st.subheader("📌 Project Summary")

    st.write(
        """
        The Employee Attrition Risk Scoring System uses machine
        learning to estimate the probability that an employee
        may leave the organization.

        The system provides employee-level risk scores,
        risk categories, department-level analysis,
        feature importance, and what-if scenario analysis.

        These predictions are intended to support HR decision
        making and proactive employee retention planning.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "🎯 **Objective**\n\n"
            "Identify employees with elevated predicted "
            "attrition risk."
        )

    with col2:
        st.info(
            "📊 **Analytics**\n\n"
            "Analyze attrition patterns across employees, "
            "departments and roles."
        )

    with col3:
        st.info(
            "💡 **Decision Support**\n\n"
            "Provide insights that can support proactive "
            "retention strategies."
        )
    st.markdown("---")

    st.subheader("⬇️ Download Risk Analysis")

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Employee Risk Data",
        data=csv_data,
        file_name="employee_attrition_risk_analysis.csv",
        mime="text/csv"
    )
    st.markdown("---")

    st.subheader("💼 HR Retention Recommendations")

    st.write(
        "The following recommendations can be considered "
        "when reviewing employees with elevated predicted risk:"
    )

    recommendations = [
        "Review workload and overtime among high-risk employees.",
        "Conduct regular employee satisfaction and engagement surveys.",
        "Provide career development and promotion opportunities.",
        "Review work-life balance for employees showing elevated risk."
      ]

    for recommendation in recommendations:
        st.write(f"• {recommendation}")

    # --------------------------------------------------------
    # KPI VALUES
    # --------------------------------------------------------

    total_employees = len(filtered_df)

    high_risk = len(
        filtered_df[
            filtered_df["RiskCategory"] == "High Risk"
        ]
    )

    medium_risk = len(
        filtered_df[
            filtered_df["RiskCategory"] == "Medium Risk"
        ]
    )

    low_risk = len(
        filtered_df[
            filtered_df["RiskCategory"] == "Low Risk"
        ]
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Employees",
        total_employees
    )

    col2.metric(
        "🔴 High Risk",
        high_risk
    )

    col3.metric(
        "🟡 Medium Risk",
        medium_risk
    )

    col4.metric(
        "🟢 Low Risk",
        low_risk
    )

    st.markdown("---")

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📈 Risk Distribution")

    risk_counts = (
        filtered_df["RiskCategory"]
        .value_counts()
        .reindex(
            ["Low Risk", "Medium Risk", "High Risk"],
            fill_value=0
        )
    )

    st.bar_chart(risk_counts)

    # --------------------------------------------------------
    # HIGH-RISK EMPLOYEES
    # --------------------------------------------------------

    st.subheader("🔴 High-Risk Employees")

    high_risk_table = (
        filtered_df[
            filtered_df["RiskCategory"] == "High Risk"
        ]
        .sort_values(
            "RiskScore",
            ascending=False
        )
    )

    if len(high_risk_table) > 0:

        st.dataframe(
            high_risk_table[
                [
                    "EmployeeID",
                    "Department",
                    "JobRole",
                    "RiskScore",
                    "RiskCategory",
                    "OverTime",
                    "JobSatisfaction",
                    "YearsAtCompany"
                ]
            ],
            use_container_width=True
        )

    else:

        st.info(
            "No high-risk employees match the selected filters."
        )

# ============================================================
# EMPLOYEE RISK PROFILE
# ============================================================

elif page == "Employee Risk Profile":

    st.header("👤 Employee Risk Profile")

    # Only show employees matching filters
    employee_ids = filtered_df[
        "EmployeeID"
    ].tolist()

    if len(employee_ids) == 0:

        st.warning(
            "No employees match the selected filters."
        )

    else:

        selected_employee = st.selectbox(
            "Select Employee ID",
            employee_ids
        )

        employee = filtered_df[
            filtered_df["EmployeeID"] == selected_employee
        ].iloc[0]

        st.subheader(
            f"Employee: {selected_employee}"
        )

        # ----------------------------------------------------
        # EMPLOYEE KPIs
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Attrition Probability",
            f"{employee['RiskScore']:.2f}%"
        )

        col2.metric(
            "Risk Category",
            employee["RiskCategory"]
        )

        col3.metric(
            "Department",
            employee["Department"]
        )

        st.markdown("---")

        # ----------------------------------------------------
        # EMPLOYEE DETAILS
        # ----------------------------------------------------

        st.subheader("📋 Employee Details")

        details = pd.DataFrame({
            "Attribute": [
                "Job Role",
                "Age",
                "Business Travel",
                "Overtime",
                "Job Satisfaction",
                "Environment Satisfaction",
                "Job Involvement",
                "Work-Life Balance",
                "Monthly Income",
                "Years at Company",
                "Years in Current Role",
                "Years Since Last Promotion",
                "Years With Current Manager"
            ],
            "Value": [
                employee["JobRole"],
                employee["Age"],
                employee["BusinessTravel"],
                employee["OverTime"],
                employee["JobSatisfaction"],
                employee["EnvironmentSatisfaction"],
                employee["JobInvolvement"],
                employee["WorkLifeBalance"],
                employee["MonthlyIncome"],
                employee["YearsAtCompany"],
                employee["YearsInCurrentRole"],
                employee["YearsSinceLastPromotion"],
                employee["YearsWithCurrManager"]
            ]
        })

        st.table(details)

# ------------------------------------------
# DEPARTMENT-LEVEL RISK ANALYSIS
# ------------------------------------------

elif page == "Department Risk Analysis":

    st.subheader("🏢 Department-Level Risk Analysis")
    st.write("📊 Department Risk Summary")

    department_summary = (
        filtered_df.groupby("Department")
        .agg(
            Employees=("EmployeeID", "count"),
            AverageRisk=("RiskScore", "mean"),
            HighRiskEmployees=("RiskCategory", lambda x: (x == "High Risk").sum())
        )
        .reset_index()
    )

    department_summary["AverageRisk"] = (
        department_summary["AverageRisk"].round(2)
    )

    header1, header2, header3, header4, header5 = st.columns(
    [2.5, 1.2, 1.2, 1.5, 1]
    )

    header1.write("**Department**")
    header2.write("**Employees**")
    header3.write("**Average Risk**")
    header4.write("**High Risk Employees**")
    header5.write("**View**")
    
    for index, row in department_summary.iterrows():

        col1, col2, col3, col4, col5 = st.columns(
            [2.5, 1.2, 1.2, 1.5, 1]
        )

        col1.write(row["Department"])
        col2.write(int(row["Employees"]))
        col3.write(row["AverageRisk"])
        col4.write(int(row["HighRiskEmployees"]))

        if col5.button(
            "View",
            key=f"view_{row['Department']}"
        ):

            department_employees = filtered_df[
                filtered_df["Department"] == row["Department"]
            ]

            st.markdown(
                f"### 👥 Employees in {row['Department']}"
            )

            display_columns = [
                "EmployeeID",
                "Department",
                "JobRole",
                "MonthlyIncome",
                "JobSatisfaction",
                "OverTime",
                "YearsAtCompany",
                "AttritionProbability",
                "RiskCategory"
            ]

            display_columns = [
                col for col in display_columns
                if col in department_employees.columns
            ]

            st.dataframe(
                department_employees[display_columns],
                use_container_width=True,
                hide_index=True
            )


# ==========================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.header("🔍 Model Explainability")

    st.write(
        "This section explains the factors associated "
        "with employee attrition risk using the "
        "Logistic Regression model."
    )

    # --------------------------------------------------------
    # RISK CATEGORIES
    # --------------------------------------------------------

    st.subheader("🎯 Risk Categories")

    risk_info = pd.DataFrame({
        "Risk Category": [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        "Probability Range": [
            "Below 30%",
            "30% – 60%",
            "Above 60%"
        ]
    })

    st.table(risk_info)

    st.markdown("---")

    # --------------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("📊 Feature Importance")

    try:

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        coefficients = classifier.coef_[0]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Importance": abs(coefficients)
        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)
        )

        st.write(
            "The chart below shows the 15 features "
            "with the strongest influence on the model."
        )

        chart_data = (
            importance_df
            .set_index("Feature")
            ["Importance"]
        )

        st.bar_chart(chart_data)

        st.dataframe(
            importance_df[
                [
                    "Feature",
                    "Coefficient",
                    "Importance"
                ]
            ],
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Feature importance could not be displayed: {e}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # INDIVIDUAL EMPLOYEE EXPLANATION
    # --------------------------------------------------------

    st.subheader("👤 Individual Employee Explanation")

    explanation_ids = filtered_df[
        "EmployeeID"
    ].tolist()

    if len(explanation_ids) == 0:

        st.warning(
            "No employees match the selected filters."
        )

    else:

        selected_explanation_employee = st.selectbox(
            "Select Employee",
            explanation_ids,
            key="explanation_employee"
        )

        employee = filtered_df[
            filtered_df["EmployeeID"]
            == selected_explanation_employee
        ].iloc[0]

        # ----------------------------------------------------
        # EMPLOYEE RISK
        # ----------------------------------------------------

        probability = employee["RiskScore"]

        if probability > 60:

            st.error(
                f"🔴 High Risk — {probability:.2f}%"
            )

        elif probability >= 30:

            st.warning(
                f"🟡 Medium Risk — {probability:.2f}%"
            )

        else:

            st.success(
                f"🟢 Low Risk — {probability:.2f}%"
            )

        # ----------------------------------------------------
        # REASON CODES
        # ----------------------------------------------------

        st.subheader("💡 Possible Contributing Factors")

        reasons = []

        if employee["OverTime"] == "Yes":
            reasons.append(
                "Overtime: Employee works overtime."
            )

        if employee["JobSatisfaction"] <= 2:
            reasons.append(
                "Job Satisfaction: Satisfaction level is low."
            )

        if employee["EnvironmentSatisfaction"] <= 2:
            reasons.append(
                "Environment Satisfaction: "
                "Work environment satisfaction is low."
            )

        if employee["WorkLifeBalance"] <= 2:
            reasons.append(
                "Work-Life Balance: "
                "Work-life balance is relatively low."
            )

        if employee["JobInvolvement"] <= 2:
            reasons.append(
                "Job Involvement: "
                "Employee involvement level is relatively low."
            )

        if employee["YearsSinceLastPromotion"] >= 5:
            reasons.append(
                "Promotion Delay: "
                "Employee has spent a long period since the last promotion."
            )

        if employee["DistanceFromHome"] >= 15:
            reasons.append(
                "Distance From Home: "
                "Employee lives relatively far from the workplace."
            )

        if employee["YearsWithCurrManager"] >= 7:
            reasons.append(
                "Manager Tenure: "
                "Employee has worked with the current manager for many years."
            )

        if employee["NumCompaniesWorked"] >= 5:
            reasons.append(
                "Previous Employment: "
                "Employee has worked for several previous companies."
            )

        if employee["YearsAtCompany"] <= 2:
            reasons.append(
                "Company Tenure: "
                "Employee has relatively short tenure at the company."
            )

        if len(reasons) == 0:

            st.info(
                "No major rule-based risk factors were identified "
                "for this employee."
            )

        else:

            for reason in reasons:
                st.write(f"• {reason}")

        # ----------------------------------------------------
        # EMPLOYEE INFORMATION
        # ----------------------------------------------------

        st.subheader("📋 Selected Employee Information")

        employee_info = pd.DataFrame({
            "Attribute": [
                "Employee ID",
                "Department",
                "Job Role",
                "Risk Score",
                "Risk Category",
                "Overtime",
                "Job Satisfaction",
                "Environment Satisfaction",
                "Work-Life Balance",
                "Years at Company",
                "Years Since Last Promotion",
                "Distance From Home"
            ],
            "Value": [
                employee["EmployeeID"],
                employee["Department"],
                employee["JobRole"],
                f"{employee['RiskScore']:.2f}%",
                employee["RiskCategory"],
                employee["OverTime"],
                employee["JobSatisfaction"],
                employee["EnvironmentSatisfaction"],
                employee["WorkLifeBalance"],
                employee["YearsAtCompany"],
                employee["YearsSinceLastPromotion"],
                employee["DistanceFromHome"]
            ]
        })

        st.table(employee_info)

    st.markdown("---")

    # --------------------------------------------------------
    # TOP RISK EMPLOYEES
    # --------------------------------------------------------

    st.subheader("🔴 Top Risk Employees")

    top_risk = (
        filtered_df
        .sort_values(
            "RiskScore",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        top_risk[
            [
                "EmployeeID",
                "Department",
                "JobRole",
                "RiskScore",
                "RiskCategory",
                "OverTime",
                "JobSatisfaction",
                "WorkLifeBalance",
                "YearsSinceLastPromotion"
            ]
        ],
        use_container_width=True
    )
# ============================================================
# WHAT-IF SCENARIO ANALYSIS
# ============================================================

elif page == "What-If Analysis":

    st.header("🎚️ What-If Scenario Analysis")

    st.write(
        "Change selected employee factors and observe how "
        "the predicted attrition risk changes."
    )

    st.info(
        "This tool is for scenario exploration. "
        "Changing a factor does not mean that it causes attrition."
    )

    # --------------------------------------------------------
    # SELECT EMPLOYEE
    # --------------------------------------------------------

    employee_ids = filtered_df["EmployeeID"].tolist()

    if len(employee_ids) == 0:

        st.warning("No employees match the selected filters.")

    else:

        selected_employee = st.selectbox(
            "Select Employee",
            employee_ids,
            key="what_if_employee"
        )

        employee = filtered_df[
            filtered_df["EmployeeID"] == selected_employee
        ].iloc[0]

        st.subheader("👤 Current Employee Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Current Risk",
                f"{employee['RiskScore']:.2f}%"
            )

        with col2:
            st.metric(
                "Department",
                employee["Department"]
            )

        with col3:
            st.metric(
                "Job Role",
                employee["JobRole"]
            )

        st.markdown("---")

        # ----------------------------------------------------
        # CURRENT VALUES
        # ----------------------------------------------------

        st.subheader("⚙️ Adjust Employee Factors")

        col1, col2 = st.columns(2)

        with col1:

            overtime = st.selectbox(
                "OverTime",
                ["Yes", "No"],
                index=(
                    0
                    if employee["OverTime"] == "Yes"
                    else 1
                )
            )

            job_satisfaction = st.slider(
                "Job Satisfaction",
                1,
                4,
                int(employee["JobSatisfaction"])
            )

            environment_satisfaction = st.slider(
                "Environment Satisfaction",
                1,
                4,
                int(employee["EnvironmentSatisfaction"])
            )

            work_life_balance = st.slider(
                "Work-Life Balance",
                1,
                4,
                int(employee["WorkLifeBalance"])
            )

        with col2:

            monthly_income = st.number_input(
                "Monthly Income",
                min_value=1000,
                max_value=100000,
                value=int(employee["MonthlyIncome"]),
                step=500
            )

            years_at_company = st.number_input(
                "Years At Company",
                min_value=0,
                max_value=50,
                value=int(employee["YearsAtCompany"]),
                step=1
            )

            years_since_promotion = st.number_input(
                "Years Since Last Promotion",
                min_value=0,
                max_value=20,
                value=int(employee["YearsSinceLastPromotion"]),
                step=1
            )

        # ----------------------------------------------------
        # CREATE WHAT-IF DATA
        # ----------------------------------------------------

        what_if_employee = employee.copy()

        what_if_employee["OverTime"] = overtime
        what_if_employee["JobSatisfaction"] = job_satisfaction
        what_if_employee[
            "EnvironmentSatisfaction"
        ] = environment_satisfaction

        what_if_employee[
            "WorkLifeBalance"
        ] = work_life_balance

        what_if_employee[
            "MonthlyIncome"
        ] = monthly_income

        what_if_employee[
            "YearsAtCompany"
        ] = years_at_company

        what_if_employee[
            "YearsSinceLastPromotion"
        ] = years_since_promotion

        # ----------------------------------------------------
        # PREPARE MODEL INPUT
        # ----------------------------------------------------

        try:

            model_features = [
                col
                for col in df.columns
                if col != "Attrition"
                and col != "EmployeeID"
            ]

            what_if_input = pd.DataFrame(
                [what_if_employee]
            )

            what_if_input = what_if_input[
                model_features
            ]

            # ------------------------------------------------
            # PREDICT NEW RISK
            # ------------------------------------------------

            new_probability = (
                model.predict_proba(
                    what_if_input
                )[0][1]
                * 100
            )

            original_probability = (
                float(employee["RiskScore"])
            )

            difference = (
                new_probability
                - original_probability
            )

            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("📊 What-If Result")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Original Risk",
                    f"{original_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Scenario Risk",
                    f"{new_probability:.2f}%"
                )

            with col3:

                st.metric(
                    "Risk Change",
                    f"{difference:+.2f}%"
                )

            # ------------------------------------------------
            # RISK CATEGORY
            # ------------------------------------------------

            if new_probability > 60:

                new_category = "High Risk"
                st.error(
                    f"🔴 Scenario Result: {new_category}"
                )

            elif new_probability >= 30:

                new_category = "Medium Risk"
                st.warning(
                    f"🟡 Scenario Result: {new_category}"
                )

            else:

                new_category = "Low Risk"
                st.success(
                    f"🟢 Scenario Result: {new_category}"
                )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.subheader("💡 Scenario Interpretation")

            if difference > 0:

                st.warning(
                    f"The modified scenario increases the "
                    f"predicted attrition risk by "
                    f"{difference:.2f} percentage points."
                )

            elif difference < 0:

                st.success(
                    f"The modified scenario decreases the "
                    f"predicted attrition risk by "
                    f"{abs(difference):.2f} percentage points."
                )

            else:

                st.info(
                    "The predicted risk remains unchanged."
                )

            # ------------------------------------------------
            # COMPARISON TABLE
            # ------------------------------------------------

            st.subheader("🔄 Before vs Scenario")

            comparison = pd.DataFrame({

                "Factor": [
                    "OverTime",
                    "Job Satisfaction",
                    "Environment Satisfaction",
                    "Work-Life Balance",
                    "Monthly Income",
                    "Years At Company",
                    "Years Since Last Promotion"
                ],

                "Original Value": [
                    employee["OverTime"],
                    employee["JobSatisfaction"],
                    employee["EnvironmentSatisfaction"],
                    employee["WorkLifeBalance"],
                    employee["MonthlyIncome"],
                    employee["YearsAtCompany"],
                    employee["YearsSinceLastPromotion"]
                ],

                "Scenario Value": [
                    overtime,
                    job_satisfaction,
                    environment_satisfaction,
                    work_life_balance,
                    monthly_income,
                    years_at_company,
                    years_since_promotion
                ]
            })

            st.dataframe(
                comparison,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Unable to calculate scenario prediction: {e}"
            )
# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Machine Learning–Based Employee Attrition Prediction "
    "and Risk Scoring System"
)
