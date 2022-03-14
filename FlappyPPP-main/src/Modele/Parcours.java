package Modele;

import java.awt.*;
import java.util.ArrayList;
import java.util.Random;


/** Parcours
 *La classe Parcours permet la création de la ligne de parcours du jeu (cela est en 2ème partie et a été fais en avance)
 */


public class Parcours extends Thread {
    private static final int step = 50;
    private final Random random = new Random();
    private Etat etat;

    private int LARGEUR_FENETRE;
    public static ArrayList<Point> parcours = new ArrayList<>();

    public static int pos = 0;
    
    public Parcours(Etat e) {
    	etat = e;
    	
    }

    public void startParcours(int largeur, int hauteur) {
        LARGEUR_FENETRE = largeur;
        pos = 1;

        Point p = new Point(0, hauteur);

        parcours.add(p);
        for (int i = 1; i < 21; i++) {
            int length = 30 + random.nextInt(step);
            int new_y = p.y + (int) ((random.nextDouble() * 2 - 1) * length);
            parcours.add(new Point(p.x + length, Math.max(new_y, 0)));
            p = parcours.get(i);
        }

        start();
    }


    /**
     * Permet l'actualisation du parcours (non finis) pour avoir un chemin infini
     * @param /
     * @return void
     */


    @Override
    public void run() {
        while (etat.testPerdu()) {
            try {
                Thread.sleep(500);

                Point last = parcours.get(parcours.size() - 1);
                if (last.x < LARGEUR_FENETRE - 30) {
                    int lastY = last.y;
                    int lastX = last.x;
                    for (int i = 0; i < 2; i++) {
                        int length = 30 + random.nextInt(step);
                        int new_y = lastY + (int) ((random.nextDouble() * 2 - 1) * length);

                        lastX += length;
                        lastY = Math.max(new_y, 0);

                        parcours.add(new Point(lastX, lastY));
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


    /**
     * Getter Parcours
     * @param /
     * @return ArrayList parcours
     */


    public ArrayList<Point> getParcours() {
        if (parcours.get(2).x < 0) {
            parcours = new ArrayList<>(parcours.subList(2, parcours.size()));
        }
        return parcours;
    }


    /**
     * Getter Position
     * @param /
     * @return int pos
     */


    public int getPosition() {
        return pos/step;
    }
   
    public static void setPosition() {
        pos+=1;
        }
    }



