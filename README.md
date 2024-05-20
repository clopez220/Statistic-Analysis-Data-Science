### Contexto empresarial
El advenimiento de los portales de noticias electrónicas nos ha ofrecido una gran oportunidad para obtener rápidamente actualizaciones sobre los eventos diarios que ocurren a nivel mundial. La información en estos portales se recupera electrónicamente de bases de datos en línea, se procesa utilizando una variedad de software y luego se transmite a los usuarios. Hay múltiples ventajas de transmitir noticias electrónicamente, como un acceso más rápido al contenido y la capacidad de utilizar diferentes tecnologías como audio, gráficos, video y otros elementos interactivos que no se utilizan o aún no son comunes en los periódicos tradicionales.

E-news Express, un portal de noticias en línea, tiene como objetivo expandir su negocio mediante la adquisición de nuevos suscriptores. Dado que cada visitante del sitio web realiza ciertas acciones en función de su interés, la compañía planea analizar estas acciones para comprender los intereses de los usuarios y determinar cómo impulsar un mejor compromiso. Los ejecutivos de E-news Express opinan que ha habido una disminución en los nuevos suscriptores mensuales en comparación con el año pasado porque la página web actual no está lo suficientemente bien diseñada en términos de esquema y contenido recomendado para mantener a los clientes comprometidos el tiempo suficiente para tomar la decisión de suscribirse.

[Las empresas suelen analizar las respuestas de los usuarios a dos variantes de un producto para decidir cuál de las dos variantes es más eficaz. Esta técnica experimental, conocida como prueba A/B, se utiliza para determinar si una nueva función atrae a los usuarios en función de una métrica elegida.]

Objetivo
El equipo de diseño de la empresa investigó y creó una nueva página de destino que tiene un nuevo esquema y muestra contenido más relevante en comparación con la página anterior. Para probar la eficacia de la nueva página de destino para reunir nuevos suscriptores, el equipo de ciencia de datos realizó un experimento seleccionando al azar a 100 usuarios y dividiéndolos en dos grupos por igual. La página de destino existente se mostró al primer grupo (grupo de control) y la nueva página de destino al segundo grupo (grupo de tratamiento). Se recopilaron datos sobre la interacción de los usuarios de ambos grupos con las dos versiones de la página de destino. Como científico de datos en E-news Express, se le ha pedido que explore los datos y realice un análisis estadístico (a un nivel de significación del 5 %) para determinar la eficacia de la nueva página de destino para reunir nuevos suscriptores para el portal de noticias. respondiendo las siguientes preguntas:

¿Los usuarios pasan más tiempo en la nueva página de destino que en la página de destino existente?

¿La tasa de conversión (la proporción de usuarios que visitan la página de destino y se convierten) de la página nueva es mayor que la tasa de conversión de la página anterior?

¿El estado convertido depende del idioma preferido? [Sugerencia: cree una tabla de contingencia usando la función pandas.crosstab()]

¿El tiempo dedicado a la nueva página es el mismo para los diferentes usuarios de idiomas?

### Diccionario de datos
Los datos contienen información sobre la interacción de los usuarios de ambos grupos con las dos versiones de la página de destino.

user_id - ID de usuario único de la persona que visita el sitio web

grupo: si el usuario pertenece al primer grupo (control) o al segundo grupo (tratamiento)

landing_page: si la página de destino es nueva o antigua

tiempo_pasado_en_la_página: tiempo (en minutos) pasado por el usuario en la página de destino

convertido: si el usuario se convierte en suscriptor del portal de noticias o no

language_preferred: idioma elegido por el usuario para ver la página de destino

## Analisis Noti-Express
Se presentan las siguientes conclusiones y recomendaciones están basadas en un análisis estadístico robusto y deberían guiar futuras decisiones estratégicas para mejorar la interacción y conversión en el sitio web.
Implementar definitivamente la nueva página.
Investigar elementos específicos que contribuyen al aumento en la tasa de conversión.
Monitorear continuamente las tasas de conversión por idioma.
Analizar factores adicionales para optimizar la experiencia del usuario según el idioma.

* Implementar definitivamente la nueva página.
* Investigar elementos específicos que contribuyen al aumento en la tasa de conversión.
* Monitorear continuamente las tasas de conversión por idioma.
* Analizar factores adicionales para optimizar la experiencia del usuario según el idioma.
