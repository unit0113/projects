using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class BulletBehavior : MonoBehaviour
{
    Rigidbody2D body;
    PlayerMovement player;
    [SerializeField] float bulletSpeed;
    void Start()
    {
        body = GetComponent<Rigidbody2D>();
        player = FindObjectOfType<PlayerMovement>();
        body.velocity = new Vector2(bulletSpeed * player.transform.localScale.x, 0f);
    }


    void Update()
    {
        
    }

    void OnTriggerEnter2D(Collider2D other) {
        if (other.tag == "Enemy") {
            Destroy(other.gameObject);
        }
        Destroy(gameObject);
    }

    void OnCollisionEnter2D() {
        Destroy(gameObject);
    }
}
