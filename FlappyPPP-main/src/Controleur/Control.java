package Controleur;

import Modele.Etat;
import Vue.Affichage;

import javax.swing.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;


/** Control
 *La classe Control implémente un (ou plusieurs) listeners (dans notre cas, le listener MouseListener).
 * Ce listener fait appels à la méthode jump de la classe Etat pour modifier les variables du modèle.
 * Ensuite, il informe la vue d’un changement (par exemple en utilisant directement la méthode repaint de la classe Affichage).
 * Cette classe dispose donc a minima d’un attribut de type Etat et d’un autre de type Affichage.
 */

public class Control implements MouseListener, KeyListener {
    private final Etat etat;
    private final Affichage vue;

    public Control(Etat e, Affichage a) {
        etat = e;
        vue = a;
    }


    /**
     * Permet l'actualisation memoire de l'affichage et de la position de l'ovale (futur oiseau)
     * @param e
     * @return void
     */


    @Override
    public void mouseClicked(MouseEvent e) {
        if (SwingUtilities.isLeftMouseButton(e)) {
            etat.jump();
            vue.change();
        }
    }


    /**
     * Toutes les interactions possibles.
     * @param e
     * @return void
     */


    @Override
    public void mousePressed(MouseEvent e) {
    }

    @Override
    public void mouseReleased(MouseEvent e) {
    }

    @Override
    public void mouseEntered(MouseEvent e) {
    }

    @Override
    public void mouseExited(MouseEvent e) {
    }

    @Override
    public void keyTyped(KeyEvent e) {

    }

    @Override
    public void keyPressed(KeyEvent e) {
        etat.jump();
        vue.change();
    }

    @Override
    public void keyReleased(KeyEvent e) {

    }
}
