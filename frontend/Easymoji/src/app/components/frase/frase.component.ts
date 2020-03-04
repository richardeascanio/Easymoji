import { Component, OnInit } from '@angular/core';
import { FraseService } from 'src/app/services/frase.service';
import { Frase } from 'src/app/models/Frase';

@Component({
  selector: 'app-frase',
  templateUrl: './frase.component.html',
  styleUrls: ['./frase.component.css']
})
export class FraseComponent implements OnInit {

  frase: Frase = {
    texto: ''
  };

  // booleano para mostrar la cajita con los resultados
  isShown = false;
  // booleano para mostrar el drawable de cargando
  progress = false;
  // booleano para mostrar el alert de error
  showDanger = false;

  // listas para guardar los emojis obtenidos de la peticion junto con su porcentaje
  emojis = [];
  porcentaje = [];

  // inyectamos el servicio al componente para poder ejecutar el motodo que tiene la peticion al back
  constructor(private fraseService: FraseService) { }

  ngOnInit() {
  }

  saveFrase() {

    // inicializamos las variables locales
    this.isShown = false;
    this.progress = true;
    this.showDanger = false;
    this.emojis = [];
    this.porcentaje = [];

    // llamamos al metodo del servicio para hacer la peticion con la frase que ingreso el usuario en el input
    this.fraseService.addFrase(this.frase).subscribe((vector) => {
      console.log('response', vector);
      // nos subscribimos al metodo ya que devuelve un observable, esperamos su respuesta (los emojis)
      // como los emojis y los porcentajes vienen en el mismo vector, los separamos en dos listas
      for (let i = 0; i < 6; i++) {
        if (i % 2 === 0) {
          this.emojis.push(vector[i]);
        } else {
          this.porcentaje.push(vector[i]);
        }
      }
      console.log('emojis', this.emojis);
      console.log('porcentaje', this.porcentaje);
      // ocultamos el progress y mostramos los resultados
      this.isShown = true;
      this.progress = false;
    },
    err => {
      console.log(err);
      this.progress = false;
      this.showDanger = true;
    });
  }

}
