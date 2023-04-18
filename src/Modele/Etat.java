package Modele;

import Vue.Affichage;

import java.awt.*;
import java.util.ArrayList;

/** Etat
 * La classe Etat définit une variable hauteur ainsi qu’une méthode d’accès getHauteur et une méthode de modification jump.
 * Cette dernière permet d’augmenter la valeur de la hauteur, tout en restant bornée par les dimensions définies dans la classe Affichage.
 */


public class Etat {
    final private Affichage AFF;
    public static final int SAUT = 10;
    public static final int X_OVAL = 10;
    public static final int GRAV = 5;
    private final int x;
    public static int hauteur;
    public boolean gameEnd = false;

    private Affichage game;

    private int MAX_Y;
    private int Y_OVAL = 400;
    private boolean perdu = false;

    public static Parcours parcours;

    public Etat(Affichage Af) {
        parcours = new Parcours(this);
        AFF = Af;
        this.game = Af;
        this.x = 20 - (game.WIDTH / 2);
        this.hauteur = (game.HAUTEUR_FENETRE / 2) - (game.HEIGHT / 2);
    }


    /**
     * Permet le deplacement vers le bas de l'ovale
     * @param /
     * @return void
     */


    public void moveDown() {
        Y_OVAL += GRAV;
        //if (Y_OVAL < 0) Y_OVAL = 0;
        if (Y_OVAL > MAX_Y) Y_OVAL = MAX_Y;
        AFF.revalidate();
        AFF.repaint();
    }


    /**
     * Permet le deplacement vers le haut de l'ovale
     * @param /
     * @return void
     */


    public void jump() {
        Y_OVAL -= SAUT;
        //if (Y_OVAL > MAX_Y) Y_OVAL = MAX_Y;
        if (Y_OVAL < 0) Y_OVAL = 0;
    }


    /**
     * Getter parcours
     * @param /
     * @return ArrayList
     */


    public static ArrayList<Point> getParcours() {
        return parcours.getParcours();
    }


    /**
     * Renvoie la hauteur de l'ovale
     * @param /
     * @return int Y_OVAL
     */


    public int getHauteur() {
        return Y_OVAL;
    }


    /**
     * Debut de methode pour le statut de la partie : Gagner, perdu ou en cours /!\à ne pas prendre en compte/!\
     * @param /
     * @return boolean perdu
     */
    public boolean testPerdu() {
    	 ArrayList<Point> points = this.getParcours();
        Point p0 = points.get(0);
        Point p1 = points.get(1);
        System.out.println("p0 : " + " " + p0 + " " + "p1 : " + " " + p1);
        double x = Affichage.LARG_OVAL + Affichage.WIDTH/2;
        double haut = this.hauteur;
        for (int i = 0; i < points.size()-1; i++) {
            if (points.get(i).x <= x && points.get(i+1).x >= x) {
                p0 = points.get(i);
                p1 = points.get(i+1);
                System.out.println("Premiere boucle : p0" + " " + p0 + " " + "p1 : " + " " + p1);
            }
        }
        double pente = (p1.y - p0.y)*1.0/(p1.x - p0.x);
        double b = (p1.x*p0.y - p0.x*p1.y)*1.0/(p1.x-p0.x);
        double y = pente * x + b;
        System.out.println("pente : " + pente + " " + "b : " + b + " " + "y : " + y + " " + "haut : " + haut + " " + "Affichage HEIGHT :" + Affichage.HEIGHT);
        if (y <= Y_OVAL - Affichage.HAUT_OVAL/2 || y >= Y_OVAL + Affichage.HAUT_OVAL/2) {
            return false;
        }
        return true;
    }
    
    public boolean getPerdu() {
    	return testPerdu();
    }



    /**
     * Attribut un Y maximum pour rester dans la fenêtre
     * @param MAX_Y
     * @return void
     */


    public void setMax(int MAX_Y) {
        this.MAX_Y = MAX_Y;
    }


    /**
     * Commencement de la ligne du parcours
     * @param largeur
     * @param hauteur
     * @return void
     */


    public void startParcours(int largeur, int hauteur) {
        parcours.startParcours(largeur, hauteur);
    }
}