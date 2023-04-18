package Controleur;

import Vue.Affichage;
import Modele.Parcours;

import java.awt.*;

import Modele.Parcours;


/** Control
 *La classe Avancer implémente le run permettant le developpement d'une ligne brisée infinie
 */


public class Avancer extends Thread {
    public Affichage affichage;

    public Avancer(Affichage a) {
        affichage = a;
        start();
    }

    /**
     *Permet la ligne infinie
     * @param /
     * return void
     */
    @Override
    public void run() {
        while (true) {
            try {
                Thread.sleep(16);
                Point[] points = affichage.etat.getParcours().toArray(new Point[0]);
                for (Point point : points) point.x--;
                Parcours.setPosition();
                System.out.println(Parcours.pos);
        

            } catch (Exception ignored) {
            }
        }
    }
}