using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScenePersit : MonoBehaviour
{
    void Awake() {
        int numScenePersists = FindObjectsOfType<ScenePersit>().Length;
        if (numScenePersists > 1) {
            Destroy(gameObject);
        } else {
            DontDestroyOnLoad(gameObject);
        }
    }

    public void ResetScenePersit() {
        Destroy(gameObject);
    }
}
