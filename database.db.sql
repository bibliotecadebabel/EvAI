BEGIN TRANSACTION;
DROP TABLE IF EXISTS "test_models";
CREATE TABLE IF NOT EXISTS "test_models" (
	"id"	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	"id_test"	INTEGER NOT NULL,
	"dna"	TEXT NOT NULL,
	"iteration"	INTEGER NOT NULL,
	"model_name"	TEXT NOT NULL,
	"model_weight" REAL NOT NULL,
	"current_time" TEXT NOT NULL,
	"current_alai_time" TEXT NOT NULL,
	"reset_dt_count" INTEGER NOT NULL,
	"type" INTEGER NOT NULL
);
DROP TABLE IF EXISTS "test_result";
CREATE TABLE IF NOT EXISTS "test_result" (
	"id"	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	"id_test"	INTEGER NOT NULL,
	"iteration"	INTEGER NOT NULL,
	"dna"	TEXT NOT NULL,
	"tangentPlane"	TEXT NOT NULL,
	"center"	INTEGER NOT NULL,
	"current_time" TEXT NOT NULL,
	"current_alai_time" TEXT NOT NULL,
	"reset_dt_count" INTEGER NOT NULL
);
DROP TABLE IF EXISTS "test";
CREATE TABLE IF NOT EXISTS "test" (
	"id"	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	"name"	TEXT NOT NULL,
	"dt"	REAL NOT NULL,
	"dt_min"	REAL NOT NULL,
	"batch_size"	INTEGER NOT NULL,
	"max_layers" INTEGER NOT NULL,
	"max_filters" INTEGER NOT NULL,
	"max_filter_dense" INTEGER NOT NULL,
	"max_kernel_dense" INTEGER NOT NULL,
	"max_pool_layer" INTEGER NOT NULL,
	"max_parents" INTEGER NOT NULL,
	"start_time" TEXT NOT NULL
);
COMMIT;
