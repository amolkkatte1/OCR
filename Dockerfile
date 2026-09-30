# ============================================================
# Stage 1: Build Spring Boot JAR
# Java 17 because pom.xml specifies java.version=17
# ============================================================

FROM maven:3.9.11-eclipse-temurin-17 AS builder

WORKDIR /app

# Copy Maven configuration
COPY OCR_Reader/pom.xml .

# Copy source code
COPY OCR_Reader/src ./src

# Build executable Spring Boot JAR
RUN mvn clean package -DskipTests


# ============================================================
# Stage 2: Run Spring Boot application
# ============================================================

FROM eclipse-temurin:17-jre

WORKDIR /app

# Copy generated JAR
COPY --from=builder /app/target/OCR_Reader-0.0.1-SNAPSHOT.jar app.jar

# Default application port
EXPOSE 8088

# Optional environment variable
ENV APP_ENV=default

# Render provides PORT dynamically.
# If PORT is not provided, use 8088.
CMD ["sh", "-c", "java -Xmx1024m -jar /app/app.jar --server.port=${PORT:-8088}"]