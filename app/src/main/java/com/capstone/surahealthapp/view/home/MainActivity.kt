package com.capstone.surahealthapp.view.home

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.capstone.surahealthapp.view.pertolonganpertama.PertolonganPertamaActivity
import com.capstone.surahealthapp.view.profile.ProfileActivity
import com.capstone.surahealthapp.R
import com.capstone.surahealthapp.view.rumahsakit.RumahSakitActivity
import com.capstone.surahealthapp.databinding.ActivityMainBinding
import com.google.android.material.bottomnavigation.BottomNavigationView

class MainActivity : AppCompatActivity() {


    lateinit var bottomNav: BottomNavigationView
    lateinit var binding: ActivityMainBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        bottomNav = binding.bottomBar
        bottomNav.setOnItemSelectedListener {
            when (it.itemId) {
                R.id.b_profil -> {
                    loadActivity(ProfileActivity()::class.java)
                    true
                }
                else -> {
                    true
                }
            }
        }

        binding.cvRumahSakit.setOnClickListener{
            startActivity(Intent(this, RumahSakitActivity::class.java))
        }
        binding.cvPertolonganPertama.setOnClickListener{
            startActivity(Intent(this, PertolonganPertamaActivity::class.java))
        }
    }

    private fun loadActivity(activityClass: Class<*>) {
        val intent = Intent(this, activityClass)
        startActivity(intent)
        finish()
    }

}