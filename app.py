import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ski_resort_CF import process_data, simulate_user_ratings, predict_user_rating
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random

def show_rating_prediction(data, resorts):
    st.header("Rating Prediction System")
    
    # Simulation parameters
    num_ratings = st.slider(
        "Pick Number of Resorts to Rate",
        min_value=5,
        max_value=50,
        value=5,
        key="num_ratings"
    )

    # use session_state to store selections
    if "resorts_to_rate" not in st.session_state:
        st.session_state.resorts_to_rate = set()

    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}
    
    # optional button to reset simulation
    if st.button("Reset Simulation"):
        st.session_state.resorts_to_rate = set()
        st.session_state.user_ratings = {}
        st.rerun()
    
    # First, select target resort
    target = st.selectbox(
        "Select the resort you want to predict a rating for",
        options=sorted(resorts),
        key="target_resort",
        placeholder="List of Resorts"
    )

    # button to generate ratings (only show if target is selected)
    if target:
        if st.button("Generate Random Ratings"):
            # Get random resorts excluding the target
            available_resorts = [r for r in resorts if r != target]
            st.session_state.resorts_to_rate = set(random.sample(available_resorts, num_ratings))
            st.session_state.user_ratings = simulate_user_ratings(data, num_ratings)
    
        if st.session_state.resorts_to_rate:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Generated Random Ratings")
                ratings_df = pd.DataFrame(
                    st.session_state.user_ratings.items(),
                    columns=['Resort', 'Rating']
                ).set_index('Resort')
                st.dataframe(ratings_df)
            
            with col2:
                st.subheader("Prediction Result")
                prediction = predict_user_rating(data, st.session_state.user_ratings, target)
                st.metric(
                    f"Predicted Rating for: {target}",
                    f"{prediction:.1f} / 5.0"
                )
            
            # add similarity vis after selections made
            st.subheader("Resort Similarities to Target")
            
            # calc similarities
            similarity_df = pd.DataFrame(
                cosine_similarity(data),
                index=data.index,
                columns=data.index
            )
            
            # get similarities between target and rated
            similarities = similarity_df.loc[target, list(st.session_state.user_ratings.keys())]
            # sort for visual
            similarities_sorted = similarities.sort_values(ascending=True) 
            
            # bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=similarities_sorted.values,
                y=similarities_sorted.index,
                orientation='h',
                text=similarities_sorted.values.round(4),
                textposition='auto',
            ))
            
            fig.update_layout(
                title=f"Similarity to {target}",
                xaxis_title="Score",
                yaxis_title="Ski Resort",
                height=500,
                showlegend=False,
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # caption
            st.caption("""
            Chart to show how similar each randomly rated resort is to the target resort using cosine similarity.
            These scores were used to calculate the weighted average rating prediction of the target.
            """)
    else:
        st.info("Please select a target resort first")

def main():
    # setup page layout!!
    st.set_page_config(layout="wide")

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["About Project", "Rating Prediction"])

    # Tab 1: Landing Page
    with tab1:
        st.title("Ski Resort Recommendation System")
        st.markdown("""
        ## DS4420 Final Project
        ### Sofia Zorich, Augusta Crow, Daniel Vahey
        We created this application to simulate a way to help users find ski resorts that match their preferences. We use collaborative filtering to predict what they would rate a resort based on ratings from other resorts.
        
        ### How our app works!
        1. Our algorithm uses the following numeric features from ski-resorts.csv:
            - Elevation difference
            - Total slope length
            - Number of lifts
            - Annual snowfall
        2. You can see a prediction of what rating a user would give to a resort based on their ratings of other resorts.
        
        ### Dataset
        Our dataset includes various ski resorts with their characteristics and features that help determine similarity between resorts.
        """)

    # Tab 2: Rating Prediction
    with tab2:
        st.title("Ski Resort Rating Prediction")
        
        # load data
        try:
            data = process_data('data/ski-resorts.csv')
            resorts = list(data.index)
        except Exception as e:
            st.error("Failed to load data. Please check file path.")
            return

        show_rating_prediction(data, resorts)

if __name__ == "__main__":
    main()
