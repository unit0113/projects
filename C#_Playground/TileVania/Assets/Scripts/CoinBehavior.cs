using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CoinBehavior : MonoBehaviour
{
    [SerializeField] int coinPoints = 100;
    [SerializeField] AudioClip coinSFX;
    bool wasCollected = false;
    void OnTriggerEnter2D(Collider2D other) {
        if (other.tag == "Player" && !wasCollected) {
            wasCollected = true;
            FindObjectOfType<GameSession>().addScore(coinPoints);
            AudioSource.PlayClipAtPoint(coinSFX, Camera.main.transform.position, 1f);
            gameObject.SetActive(false);
            Destroy(gameObject);
        }
    }
}
