# Text Similarity
# MBTI
# Career Goal
# Location
# Experience

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from src.matching.mbti_engine import (
    MBTIEngine
)

TECH_PROFESSIONS = {
    "Data Scientist",
    "ML Engineer",
    "AI Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "DevOps Engineer",
    "Cloud Engineer",
    "Cybersecurity Analyst"
}

HEALTHCARE_PROFESSIONS = {
    "Doctor",
    "Nurse",
    "Healthcare Analyst"
}

BUSINESS_PROFESSIONS = {
    "Business Analyst",
    "Product Manager",
    "Project Manager",
    "Consultant"
}

FINANCE_PROFESSIONS = {
    "Financial Analyst",
    "Investment Advisor",
    "Accountant"
}

DESIGN_PROFESSIONS = {
    "UI/UX Designer",
    "Graphic Designer",
    "Content Writer",
    "Marketing Specialist"
}

CAREER_GROUPS = {
    "AI Research": "AI",
    "Data Analytics": "AI",

    "Cloud Computing": "TECH",
    "Cybersecurity": "TECH",

    "Leadership": "BUSINESS",
    "Product Management": "BUSINESS",

    "Financial Growth": "FINANCE",

    "Healthcare Innovation": "HEALTHCARE",

    "Design Leadership": "DESIGN",

    "Startup Founder": "BUSINESS",
}

class Recommender:

    def __init__(
        self,
        users_df,
        tfidf_matrix
    ):
        self.users_df = users_df
        self.tfidf_matrix = tfidf_matrix

    def _experience_score(
        self,
        exp_a,
        exp_b
    ):

        diff = abs(exp_a - exp_b)

        if diff <= 2:
            return 100

        if diff <= 5:
            return 80

        if diff <= 10:
            return 60

        return 40

    def _location_score(
        self,
        loc_a,
        loc_b
    ):

        return 100 if loc_a == loc_b else 0

    def _career_goal_score(
        self,
        goal_a,
        goal_b
    ):
        if goal_a == goal_b:
            return 100
        
        group_a = CAREER_GROUPS.get(goal_a)
        group_b = CAREER_GROUPS.get(goal_b)
        if group_a and group_b and group_a == group_b:
            return 70
        
        return 0
    
    def _profession_score(
        self,
        profession_a,
        prefession_b
    ):
        if profession_a == prefession_b:
            return 100
        
        groups = [
            TECH_PROFESSIONS,
            BUSINESS_PROFESSIONS,
            FINANCE_PROFESSIONS,
            HEALTHCARE_PROFESSIONS,
            DESIGN_PROFESSIONS,
        ]
        for group in groups:
            if(
                profession_a in group and
                prefession_b in group
            ):
                return 70
        return 0

    def _skills_score(
        self,
        skills_a,
        skills_b
    ):
        if not isinstance(skills_a, str) or not isinstance(skills_b, str):
            return 0.0
        set_a = set(s.strip() for s in skills_a.split(",") if s.strip())
        set_b = set(s.strip() for s in skills_b.split(",") if s.strip())
        if not set_a or not set_b:
            return 0.0
        overlap = len(set_a & set_b)
        max_overlap = max(len(set_a), len(set_b))
        return (overlap / max_overlap) * 100.0

    def _networking_intent_score(
        self,
        intent_a,
        intent_b
    ):
        if not isinstance(intent_a, str) or not isinstance(intent_b, str):
            return 30.0

        compatible_pairs = {
            ("Find Mentor", "Find Mentee"),
            ("Find Mentee", "Find Mentor"),
            ("Startup Partner", "Startup Partner"),
            ("Research Collaboration", "Research Collaboration"),
            ("Professional Networking", "Professional Networking"),
            ("Team Building", "Team Building"),
            ("Knowledge Sharing", "Knowledge Sharing"),
            ("Career Growth", "Career Growth"),
        }

        pair = (intent_a, intent_b)
        reverse_pair = (intent_b, intent_a)

        if (
            pair in compatible_pairs
            or reverse_pair in compatible_pairs
        ):
            return 100.0

        if (
            intent_a == "Professional Networking"
            or intent_b == "Professional Networking"
        ):
            return 60.0

        return 30.0

    def compatibility_score(
        self,
        user_id_1,
        user_id_2
    ):

        idx1 = self.users_df[
            self.users_df["user_id"]
            == user_id_1
        ].index[0]

        idx2 = self.users_df[
            self.users_df["user_id"]
            == user_id_2
        ].index[0]

        user1 = self.users_df.iloc[idx1]
        user2 = self.users_df.iloc[idx2]

        text_similarity = (
            cosine_similarity(
                self.tfidf_matrix[idx1],
                self.tfidf_matrix[idx2]
            )[0][0]
            * 100
        )

        mbti_score = (
            MBTIEngine.get_score(
                user1["mbti"],
                user2["mbti"]
            )
        )
        profession_score =(
            self._profession_score(
                user1["profession"],
                user2["profession"]
            )
        )
        career_score = (
            self._career_goal_score(
                user1["career_goal"],
                user2["career_goal"]
            )
        )

        location_score = (
            self._location_score(
                user1["location"],
                user2["location"]
            )
        )

        experience_score = (
            self._experience_score(
                user1["experience_years"],
                user2["experience_years"]
            )
        )

        skills1 = user1.get("skills", "")
        skills2 = user2.get("skills", "")
        skills_score = (
            self._skills_score(
                skills1,
                skills2
            )
        )

        intent1 = user1.get("networking_intent", "")
        intent2 = user2.get("networking_intent", "")
        networking_intent_score = (
            self._networking_intent_score(
                intent1,
                intent2
            )
        )

        final_score = (

            0.30 * text_similarity +

            0.10 * mbti_score +
            
            0.15 * profession_score +
            
            0.10 * career_score +

            0.05 * location_score +

            0.10 * experience_score +

            0.15 * skills_score +

            0.05 * networking_intent_score
        )

        return {

            "text_similarity":
                round(text_similarity, 2),

            "mbti_score":
                round(mbti_score, 2),
                
            "profession_score":
                round(profession_score, 2),
                
            "career_goal_score":
                round(career_score, 2),

            "location_score":
                round(location_score, 2),

            "experience_score":
                round(experience_score, 2),

            "skills_score":
                round(skills_score, 2),

            "networking_intent_score":
                round(networking_intent_score, 2),

            "final_score":
                round(final_score, 2)
        }
    def get_top_recommendations(
            self,
            user_id,
            top_n=5
    ):

        recommendations = []

        for target_id in self.users_df["user_id"]:

            if target_id == user_id:
                continue

            result = self.compatibility_score(
                user_id,
                target_id
            )

            target_user = self.users_df[self.users_df["user_id"] == target_id].iloc[0]

            recommendations.append(
                {
                    "user_id":
                        target_id,

                    "profession":
                        target_user["profession"],

                    "mbti":
                        target_user["mbti"],

                    "career_goal":
                        target_user["career_goal"],

                    "location":
                        target_user["location"],

                    "experience":
                        target_user["experience_years"],

                    "final_score":
                        result["final_score"]
                }
            )

        recommendations.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )
        return recommendations[:top_n]