package model;

import model.Osoba;
import model.Banka;

public class Student extends Osoba {

	private String index;
	private double prosek;
	private Banka banka;

	public Student () { }

	public Student (String index, double prosek, Banka banka) {
		this.index = index;
		this.prosek = prosek;
		this.banka = banka;
	}


	public void podaci ( ) {
		return null;
	}

}
