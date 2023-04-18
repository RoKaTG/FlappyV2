package Controleur;

import Modele.Etat;
import Vue.Affichage;

import java.util.Random;


/** Voler
 * La classe voler permet la chute libre de l'ovale en limite de fenêtre.
 */


public class Voler extends Thread {
    final Etat etat;
    public Affichage affichage;

    public static final Random rand = new Random();

    public Voler(Etat e, Affichage a) {
        etat = e;
        affichage = a;
    }


    /**
     * Permet l'application de la gravité sur l'ovale
     * @param /
     * @return void
     */


    @Override
    public void run() {
        while (etat.testPerdu()) {
            etat.moveDown();
            System.out.println(etat.testPerdu());
            try {
                Thread.sleep(rand.nextInt(200)+100);
            }
            catch (InterruptedException e) {
                e.printStackTrace();
                }
            }
        affichage.affichageFinPartie();
        }
}

