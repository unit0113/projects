using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Health : MonoBehaviour
{
    [SerializeField] bool isPlayer;
    [SerializeField] int health = 50;
    [SerializeField] int points = 50;
    [SerializeField] ParticleSystem hitEffect;
    [SerializeField] bool applyCameraShake;
    CameraShake cameraShake;
    AudioPlayer audioPlayer;
    ScoreKeeper scoreKeeper;
    LevelManager levelManager;

    void Awake() {
        cameraShake = Camera.main.GetComponent<CameraShake>();
        audioPlayer = FindObjectOfType<AudioPlayer>();
        scoreKeeper = FindObjectOfType<ScoreKeeper>();
        levelManager = FindObjectOfType<LevelManager>();
    }

    void OnTriggerEnter2D(Collider2D other) {
        DamageDealer damageDealer = other.GetComponent<DamageDealer>();
        if (damageDealer) {
            damageDealer.Hit();
            TakeDamage(damageDealer.GetDamage());  
            PlayHitEffect(); 
            ShakeCamera();    
            audioPlayer.PlayDamageClip();     
        }
    }

    void TakeDamage(int damage) {
        health -= damage;
        if (health <= 0) {
            Die();
        }
    }

    void Die() {
        if (!isPlayer) {
            scoreKeeper.ModifyScore(points);
        } else {
            levelManager.LoadGameOver();
        }
        Destroy(gameObject);
    }

    void PlayHitEffect() {
        if (hitEffect) {
            ParticleSystem instance = Instantiate(hitEffect, transform.position, Quaternion.identity);
            Destroy(instance, instance.main.duration + instance.main.startLifetime.constantMax);
        }
    }

    void ShakeCamera() {
        if (cameraShake && applyCameraShake) {
            cameraShake.Play();
        }
    }
    
    public int GetHealth() { return health; }
}
