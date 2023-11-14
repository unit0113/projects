using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Timer : MonoBehaviour
{
    [SerializeField] float timeToAnswer = 10f;
    [SerializeField] float timeToShowCorrectAnswer = 5f;
    float timeRemaining;
    public bool isAnswering = true;
    public bool showNextQuestion = true;
    public float fillRatio;

    void Awake() {
        timeRemaining = timeToAnswer;
        isAnswering = true;
    }

    void Update()
    {
        updateTimer();
    }

    public void cancelTimer() {
        timeRemaining = 0;
    }

    void updateTimer() {
        timeRemaining -= Time.deltaTime;
        if (isAnswering) {
            if (timeRemaining <=0) {
                isAnswering = false;
                timeRemaining = timeToShowCorrectAnswer;
            } else {
                fillRatio = timeRemaining / timeToAnswer;
            }
        } else {
            if (timeRemaining <=0) {
                isAnswering = true;
                timeRemaining = timeToAnswer;
                showNextQuestion = true;
            } else {
                fillRatio = timeRemaining / timeToShowCorrectAnswer;
            }
        }
    }
}
