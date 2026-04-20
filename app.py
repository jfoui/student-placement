import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    classifier = joblib.load(BASE_DIR / 'RandomForest_Classifier_pipeline.pkl')
    regressor = joblib.load(BASE_DIR / 'LinearRegression_pipeline.pkl')
    return classifier, regressor

classifier_model, regressor_model = load_models()

def main():
    st.set_page_config(page_title="Student Placement Prediction", layout="wide")
    st.title('Student Placement & Salary Prediction')

    st.markdown("Enter student details below to predict placement status and estimated salary.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal & Academic")
        gender = st.radio('Gender', ['Male', 'Female', 'Other'])
        branch = st.selectbox('Branch', ['CSE', 'ECE', 'MECH', 'CIVIL', 'IT', 'EEE', 'OTHER'])
        cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        tenth_percentage = st.number_input('10th Percentage', min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        twelfth_percentage = st.number_input('12th Percentage', min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        backlogs = st.number_input('Active Backlogs', min_value=0, max_value=10, value=0, step=1)
        attendance_percentage = st.number_input('Attendance Percentage', min_value=0.0, max_value=100.0, value=85.0, step=1.0)

    with col2:
        st.subheader("Skills & Experience")
        projects_completed = st.number_input('Projects Completed', min_value=0, max_value=20, value=2, step=1)
        internships_completed = st.number_input('Internships Completed', min_value=0, max_value=10, value=1, step=1)
        hackathons_participated = st.number_input('Hackathons Participated', min_value=0, max_value=20, value=1, step=1)
        certifications_count = st.number_input('Certifications Count', min_value=0, max_value=20, value=2, step=1)
        coding_skill_rating = st.slider('Coding Skill Rating', min_value=1, max_value=5, value=3)
        communication_skill_rating = st.slider('Communication Skill Rating', min_value=1, max_value=5, value=3)
        aptitude_skill_rating = st.slider('Aptitude Skill Rating', min_value=1, max_value=5, value=3)

    with col3:
        st.subheader("Lifestyle & Background")
        study_hours_per_day = st.number_input('Study Hours per Day', min_value=0.0, max_value=24.0, value=4.0, step=0.5)
        sleep_hours = st.number_input('Sleep Hours', min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        stress_level = st.slider('Stress Level', min_value=1, max_value=10, value=5)
        part_time_job = st.radio('Part-time Job', ['Yes', 'No'])
        internet_access = st.radio('Internet Access', ['Yes', 'No'])
        family_income_level = st.selectbox('Family Income Level', ['Low', 'Medium', 'High'])
        city_tier = st.selectbox('City Tier', ['Tier 1', 'Tier 2', 'Tier 3'])
        extracurricular_involvement = st.selectbox('Extracurricular Involvement', ['Low', 'Medium', 'High'])

    st.markdown("---")

    if st.button('Predict Placement & Salary', use_container_width=True, type="primary"):
        # Build the input DataFrame
        df = pd.DataFrame([{
            'gender': gender,
            'branch': branch,
            'cgpa': cgpa,
            'tenth_percentage': tenth_percentage,
            'twelfth_percentage': twelfth_percentage,
            'backlogs': backlogs,
            'study_hours_per_day': study_hours_per_day,
            'attendance_percentage': attendance_percentage,
            'projects_completed': projects_completed,
            'internships_completed': internships_completed,
            'coding_skill_rating': coding_skill_rating,
            'communication_skill_rating': communication_skill_rating,
            'aptitude_skill_rating': aptitude_skill_rating,
            'hackathons_participated': hackathons_participated,
            'certifications_count': certifications_count,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'part_time_job': part_time_job,
            'family_income_level': family_income_level,
            'city_tier': city_tier,
            'internet_access': internet_access,
            'extracurricular_involvement': extracurricular_involvement
        }])

        try:
            with st.spinner("Analyzing student profile..."):
                # Predict directly using loaded models
                placement_prediction = classifier_model.predict(df)[0]
                salary_prediction = regressor_model.predict(df)[0]

            st.success("Prediction Complete!")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                status = str(placement_prediction)
                if status.lower() == "placed":
                    st.metric(label="Placement Status", value="Placed")
                else:
                    st.metric(label="Placement Status", value="Not Placed")

            with res_col2:
                salary = round(float(salary_prediction), 2)
                if status.lower() == "placed":
                    st.metric(label="Estimated Salary (LPA)", value=f"₹{salary} Lakhs")
                else:
                    st.metric(label="Estimated Salary (LPA)", value="N/A",
                              help="Salary is only predicted if the student is placed.")
            # Visualisasi sederhana
            st.markdown("---")
            st.subheader("Analisis Profil Singkat")

            # Progress bar untuk CGPA
            st.write("Skor CGPA Anda:")
            st.progress(float(cgpa) / 10.0, text=f"{cgpa} / 10.0")

            # Bar chart sederhana untuk Skill
            st.write("Distribusi Kemampuan (Skills):")
            skills_df = pd.DataFrame({
                "Skill": ["Coding", "Communication", "Aptitude"],
                "Rating (1-5)": [coding_skill_rating, communication_skill_rating, aptitude_skill_rating]
            }).set_index("Skill")

            st.bar_chart(skills_df)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
