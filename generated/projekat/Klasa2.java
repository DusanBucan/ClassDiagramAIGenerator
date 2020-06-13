package model;

import model.Klasa1;
import model.Klasa3;

public class Klasa2 extends Klasa1 {

	private Collection<Klasa3> klasa3Collection;
	public Klasa2 () { }

	public Klasa2 (Collection<Klasa3> klasa3Collection) {
		this.klasa3Collection = klasa3Collection;
	}


}
