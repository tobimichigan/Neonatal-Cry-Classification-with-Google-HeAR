from sklearn.ensemble import VotingClassifier

def build_soft_voting(models):
    return VotingClassifier(estimators=models, voting="soft")
