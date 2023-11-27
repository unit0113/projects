using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GameSession : MonoBehaviour
{
    [SerializeField] int playerLives = 3;
    [SerializeField] TextMeshProUGUI livesText;
    [SerializeField] TextMeshProUGUI scoreText;
    int score = 0;

    void Awake() {
        int numSessions = FindObjectsOfType<GameSession>().Length;
        if (numSessions > 1) {
            Destroy(gameObject);
        } else {
            DontDestroyOnLoad(gameObject);
        }
    }

    void Start() {
        livesText.text = "Lives: " + playerLives.ToString();
        scoreText.text = "Score: " + score.ToString();
    }

    public void ProcessDeath() {
        if (playerLives > 1) {
            TakeLife();
        } else {
            ResetSession();
        }
    }

    void TakeLife() {
        --playerLives;
        livesText.text = "Lives: " + playerLives.ToString();
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }

    public void addScore(int points) {
        score += points;
        scoreText.text = "Score: " + score.ToString();
    }

    void ResetSession() {
        FindObjectOfType<ScenePersit>().ResetScenePersit();
        SceneManager.LoadScene(0);
        Destroy(gameObject);
    }
}
