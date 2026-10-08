class MBTIEngine:
    """MBTI personality compatibility scoring engine.

    Uses a full 16×16 compatibility matrix derived from MBTI type dynamics.
    Scores are heuristic — MBTI is treated as a compatibility signal, not
    a scientifically validated predictor.
    """

    # Compatibility scores keyed by (type_a, type_b).
    # Based on MBTI cognitive function stacking theory:
    # - Same type: 75 (comfortable but potentially stagnant)
    # - Complementary dominant/auxiliary functions: 90-100
    # - Shared attitudes with different functions: 80-90
    # - Moderate overlap: 60-75
    # - Low natural affinity: 40-55
    # - Very different cognitive styles: 30-45
    MBTI_COMPATIBILITY = {
        "INTJ": {
            "ENFP": 100, "ENTP": 95, "INFJ": 85, "INTP": 80,
            "INTJ": 75, "ENFJ": 80, "ENTJ": 70, "INFP": 70,
            "ISTP": 60, "ESTP": 55, "ISFP": 50, "ESFP": 45,
            "ISTJ": 65, "ISFJ": 50, "ESTJ": 55, "ESFJ": 40,
        },
        "INTP": {
            "ENTJ": 100, "ENFJ": 90, "INTJ": 80, "INFJ": 80,
            "INTP": 75, "ENTP": 85, "ENFP": 75, "INFP": 70,
            "ISTP": 70, "ESTP": 60, "ISFP": 50, "ESFP": 45,
            "ISTJ": 60, "ISFJ": 50, "ESTJ": 55, "ESFJ": 40,
        },
        "ENTJ": {
            "INTP": 100, "INFP": 95, "ENTP": 85, "ENFP": 80,
            "ENTJ": 75, "INTJ": 80, "INFJ": 75, "ENFJ": 70,
            "ISTJ": 65, "ESTJ": 70, "ISTP": 60, "ESTP": 55,
            "ISFJ": 45, "ESFJ": 45, "ISFP": 50, "ESFP": 45,
        },
        "ENTP": {
            "INFJ": 100, "INTJ": 95, "ENFJ": 85, "INFP": 80,
            "ENTP": 75, "INTP": 85, "ENTJ": 80, "ENFP": 75,
            "ESTP": 65, "ISTP": 60, "ESFP": 55, "ISFP": 50,
            "ESTJ": 55, "ISTJ": 50, "ESFJ": 45, "ISFJ": 40,
        },
        "INFJ": {
            "ENTP": 100, "ENFP": 95, "INTJ": 85, "INFP": 85,
            "INFJ": 80, "INTP": 80, "ENFJ": 75, "ENTJ": 75,
            "ISFJ": 65, "ESFJ": 60, "ISFP": 65, "ESFP": 50,
            "ISTJ": 50, "ESTJ": 45, "ISTP": 55, "ESTP": 45,
        },
        "INFP": {
            "ENTJ": 95, "ENFJ": 90, "INFJ": 85, "ENFP": 85,
            "INFP": 80, "INTJ": 70, "INTP": 70, "ENTP": 80,
            "ISFP": 70, "ESFP": 60, "ISFJ": 60, "ESFJ": 50,
            "ISTJ": 45, "ESTJ": 40, "ISTP": 50, "ESTP": 45,
        },
        "ENFJ": {
            "INFP": 90, "INTP": 90, "INFJ": 75, "ENFP": 85,
            "ENFJ": 75, "INTJ": 80, "ENTJ": 70, "ENTP": 85,
            "ESFJ": 70, "ISFJ": 65, "ESFP": 65, "ISFP": 55,
            "ESTJ": 55, "ISTJ": 50, "ESTP": 50, "ISTP": 45,
        },
        "ENFP": {
            "INTJ": 100, "INFJ": 95, "ENTJ": 80, "ENTP": 75,
            "ENFP": 75, "INFP": 85, "ENFJ": 85, "INTP": 75,
            "ESFP": 70, "ISFP": 65, "ESTP": 60, "ISTP": 55,
            "ESFJ": 55, "ISFJ": 50, "ESTJ": 45, "ISTJ": 40,
        },
        "ISTJ": {
            "ESFP": 90, "ESTP": 85, "ISFP": 80, "ESTJ": 80,
            "ISTJ": 75, "ISFJ": 75, "INTJ": 65, "ENTJ": 65,
            "ISTP": 70, "INTP": 60, "ESFJ": 65, "ENFP": 40,
            "ENTP": 50, "INFJ": 50, "ENFJ": 50, "INFP": 45,
        },
        "ISFJ": {
            "ESFP": 90, "ESTP": 80, "ESTJ": 80, "ESFJ": 80,
            "ISFJ": 75, "ISTJ": 75, "ISFP": 80, "INFJ": 65,
            "ISTP": 60, "ENFP": 50, "ENTP": 40, "INTP": 50,
            "INTJ": 50, "ENTJ": 45, "ENFJ": 65, "INFP": 60,
        },
        "ESTJ": {
            "ISFP": 90, "ISTP": 85, "ESTP": 80, "ISTJ": 80,
            "ESTJ": 75, "ESFJ": 75, "ENTJ": 70, "INTJ": 55,
            "ESTP": 80, "ESFP": 65, "ISFJ": 80, "ENFP": 45,
            "ENTP": 55, "INFP": 40, "INFJ": 45, "ENFJ": 55,
        },
        "ESFJ": {
            "ISTP": 90, "ISFP": 85, "ESTP": 80, "ISTJ": 80,
            "ESFJ": 75, "ESTJ": 75, "ISFJ": 80, "ENFJ": 70,
            "ESFP": 75, "INTP": 40, "ENTP": 45, "INFP": 50,
            "INTJ": 40, "ENTJ": 45, "INFJ": 60, "ENFP": 55,
        },
        "ISTP": {
            "ESTJ": 85, "ESFJ": 90, "ESTP": 85, "ESFP": 75,
            "ISTP": 75, "ISFP": 75, "ISTJ": 70, "ISFJ": 60,
            "INTJ": 60, "INTP": 70, "ENTJ": 60, "ENTP": 60,
            "ENFP": 55, "INFP": 50, "ENFJ": 45, "INFJ": 55,
        },
        "ISFP": {
            "ESTJ": 90, "ESFJ": 85, "ISTJ": 80, "ISFJ": 80,
            "ISFP": 75, "ISTP": 75, "ESFP": 80, "ESTP": 70,
            "ENFP": 65, "INFP": 70, "ENFJ": 55, "INFJ": 65,
            "ENTJ": 50, "INTJ": 50, "ENTP": 50, "INTP": 50,
        },
        "ESTP": {
            "ISTJ": 85, "ISFJ": 80, "ESTJ": 80, "ESFJ": 80,
            "ESTP": 75, "ISTP": 85, "ESFP": 80, "ISFP": 70,
            "ENTP": 65, "ENTJ": 55, "INTJ": 55, "ENFP": 60,
            "INTP": 60, "INFP": 45, "ENFJ": 50, "INFJ": 45,
        },
        "ESFP": {
            "ISTJ": 90, "ISFJ": 90, "ESTJ": 65, "ESFJ": 75,
            "ESFP": 75, "ESTP": 80, "ISFP": 80, "ISTP": 75,
            "ENFP": 70, "INFP": 60, "ENFJ": 65, "INFJ": 50,
            "ENTJ": 45, "INTJ": 45, "ENTP": 55, "INTP": 45,
        },
    }

    @classmethod
    def get_score(cls, mbti_a: str, mbti_b: str) -> float:
        if mbti_a == mbti_b:
            return 75.0

        score = cls.MBTI_COMPATIBILITY.get(mbti_a, {}).get(mbti_b)
        if score is not None:
            return float(score)

        # Try reverse lookup
        score = cls.MBTI_COMPATIBILITY.get(mbti_b, {}).get(mbti_a)
        if score is not None:
            return float(score)

        return 50.0
