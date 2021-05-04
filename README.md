tfx-fuel-efficiency-regression
==============================

Tensorflow [fuel efficiency regression tutorial][tf_tutorial] ported to TFX
pipelines running on Apache Beam.

[tf_tutorial]: https://www.tensorflow.org/tutorials/keras/regression


Requirements
------------

- Docker (tested on 20.10.5/Community)
  - docker-compose
- `make`

Usage
-----

Via `make`:

~~~
$ make help
$ make run
~~~

Via `docker compose`:

~~~
$ docker compose pull
$ docker compose build
$ docker compose up
~~~

Components
----------

| Components                | Description                                                                      |
| ---                       | ---                                                                              |
| `postgres_example_gen`    | Query-based TFX component generating tf_examples from a Postgresql feature table |
| `mlmd_tracking_publisher` | TFX component emitting model evaluation output to mlmd database                  |
| `app`                     | Tensorflow tutorial as TFX pipeline implementation                               |
| `dashboard`               | Ad-hoc Streamlit model tracking dashboard                                        |

Model tracking dashboard
------------------------

Can be reached at http://localhost:9000 after either `make run` or `make
run-dashboard`

![dashboard](https://i.ibb.co/k9FT9Km/localhost-9000-1.png)
