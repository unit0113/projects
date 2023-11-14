using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScoreKeeper : MonoBehaviour
{
    int numCorrect = 0;
    int numQuestions = 0;

    public int getNumCorrect() {
        return numCorrect;
    }

    public int getNumQuestions() {
        return numQuestions;
    }

    public void incrementCorrect() {
        ++numCorrect;
        ++numQuestions;
    }

    public void incrementIncorrect() {
        ++numQuestions;
    }

    public int getScore() {
        return Mathf.RoundToInt(numCorrect * 100 / (float)numQuestions);
    }
}
