import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { HttpHeaders } from '@angular/common/http';
import { Frase } from '../models/Frase';

@Injectable({
  providedIn: 'root'
})
export class FraseService {

  // Url a la que nos vamos a conectar (donde se encuentra el back end)
  API_URI = 'http://127.0.0.1:8000';
  // header donde se encuentra el tipo de dato que le vamos a pasar al back end
  headers = new HttpHeaders({'Content-type': 'application/json'});

  constructor(private http: HttpClient) {}

  // Metodo donde vamos a hacer la peticion post con una frase a la direccion descrita anteriormente
  addFrase(frase: Frase): Observable<Frase> {
    return this.http.post<Frase>(`${this.API_URI}/status/`, frase);
  }
}
