using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class Quiz : MonoBehaviour {
    [Header("Questions")]
    [SerializeField] TextMeshProUGUI questionText;
     QuestionSO currentQuestion;
     [SerializeField] List<QuestionSO> questions = new List<QuestionSO>();

    [Header("Answers")]
    [SerializeField] GameObject[] answerButtons;
    bool answeredEarly = false;
    int correctAnswerIndex;

    [Header("Buttons")]
    [SerializeField] Sprite defaultAnswerSprite;
    [SerializeField] Sprite correctAnswerSprite;

    [Header("Timer")]
    [SerializeField] Image timerImage;
    Timer timer;

    [Header("Scoring")]
    [SerializeField] TextMeshProUGUI scoreText;
    ScoreKeeper scoreKeeper;

    [Header("Progress Bar")]
    [SerializeField] Slider progressBar;
    public bool complete;

    void Awake() {
        timer = FindObjectOfType<Timer>();
        scoreKeeper = FindObjectOfType<ScoreKeeper>();
        progressBar = FindObjectOfType<Slider>();
        progressBar.maxValue = questions.Count;
        progressBar.value = 0;
        complete = false;
    }

    void Update() {
        timerImage.fillAmount = timer.fillRatio;
        if (timer.showNextQuestion) {
            if (progressBar.value == progressBar.maxValue) {
                complete = true;
            }

            getNextQuestion();
            timer.showNextQuestion = false;
        } else if (!answeredEarly && !timer.isAnswering) {
            displayAnswer(-1);
            setButtonState(false);
        }

        if (!timer.isAnswering && Input.GetMouseButtonDown(0)) {
            timer.cancelTimer();
        }
    }

    public void OnAnswerSelected(int index) {
        displayAnswer(index);
        setButtonState(false);
        timer.cancelTimer();
        scoreText.text = "Score: " + scoreKeeper.getScore() + "%";
    }

    void displayAnswer(int index) {
        answeredEarly = true;
        correctAnswerIndex = currentQuestion.getCorrectAnswerIndex();
        Image buttonImage = answerButtons[correctAnswerIndex].GetComponent<Image>();
        buttonImage.sprite = correctAnswerSprite;
        if (index == correctAnswerIndex) {
            questionText.text = "Correct!";
            scoreKeeper.incrementCorrect();
        } else {
            string correctAnswer = currentQuestion.getAnswer(correctAnswerIndex);
            questionText.text = "Incorrect, the correct answer is:\n" + correctAnswer;
            scoreKeeper.incrementIncorrect();
        }
    }

    void displayQuestion() {
        questionText.text = currentQuestion.getQuestion();

        for (int i = 0; i < answerButtons.Length; ++i) {
            TextMeshProUGUI buttonText = answerButtons[i].GetComponentInChildren<TextMeshProUGUI>();
            buttonText.text = currentQuestion.getAnswer(i);
        }
    }

    void setButtonState(bool state) {
        for (int i = 0; i < answerButtons.Length; ++i) {
            Button button = answerButtons[i].GetComponent<Button>();
            button.interactable = state;
        }
    }

    void getNextQuestion() {
        answeredEarly = false;
        if (questions.Count > 0) {
            setButtonState(true);
            getRandomQuestion();
            setDefaultButtonSprites();
            displayQuestion();
            ++progressBar.value;
        }
    }

    void getRandomQuestion() {
        int index = Random.Range(0, questions.Count);
        currentQuestion = questions[index];
        questions.Remove(currentQuestion);
    }

    void setDefaultButtonSprites() {
        for (int i = 0; i < answerButtons.Length; ++i) {
            Image buttonImage = answerButtons[i].GetComponent<Image>();
            buttonImage.sprite = defaultAnswerSprite;
        }
    }
}
