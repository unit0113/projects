using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScoreKeeper : MonoBehaviour
{
    int score = 0;
    static ScoreKeeper instance;

    void Awake() {
        ManageSingleton();
    }

    void ManageSingleton() {
        if (instance) {
            gameObject.SetActive(false);
            Destroy(gameObject);
        } else {
            instance = this;
            DontDestroyOnLoad(gameObject);
        }
    }

    public int GetScore() { return score; }

    public void ModifyScore(int points) {
        score = Mathf.Max(0, score + points);
    }

    public void ResetScore() {
        score = 0;
    }
}
