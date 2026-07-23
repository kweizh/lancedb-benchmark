FROM ubuntu:24.04

ARG POCHI_VERSION

# Prevent apt from auto-starting services during image build (no systemd in Docker).
RUN printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d && chmod +x /usr/sbin/policy-rc.d

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        mysql-server mysql-client && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        pymysql==1.1.1 \
        mysql-replication==1.0.16 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Enable row-based binary logging so the CDC indexer can consume INSERT/UPDATE/DELETE
# row events with full before/after images and column-name metadata.
RUN mkdir -p /etc/mysql/mysql.conf.d && \
    printf '[mysqld]\nserver-id=1\nlog_bin=/var/lib/mysql/mysql-bin\nbinlog_format=ROW\nbinlog_row_image=FULL\nbinlog_row_metadata=FULL\nmax_binlog_size=100M\nbind-address=127.0.0.1\n' \
        > /etc/mysql/mysql.conf.d/zz-cdc.cnf

# Entrypoint: initialize the MySQL data directory if needed, start mysqld in the
# background, provision the source database/table and a replication-capable user,
# then hand control to the container CMD.
RUN cat >/usr/local/bin/mysql-entrypoint.sh <<'EOF'
#!/bin/bash
mkdir -p /var/run/mysqld && chown -R mysql:mysql /var/run/mysqld
if [ ! -d /var/lib/mysql/mysql ]; then
    mysqld --initialize-insecure --user=mysql
fi
chown -R mysql:mysql /var/lib/mysql
mysqld --user=mysql &
for i in $(seq 1 90); do
    if mysqladmin --protocol=socket -uroot ping >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
mysql --protocol=socket -uroot <<'SQL'
CREATE DATABASE IF NOT EXISTS appdb;
CREATE TABLE IF NOT EXISTS appdb.documents (
    id BIGINT PRIMARY KEY,
    title VARCHAR(255),
    body TEXT,
    category VARCHAR(64),
    price DOUBLE
);
CREATE USER IF NOT EXISTS 'cdc'@'%' IDENTIFIED WITH mysql_native_password BY 'cdcpass';
ALTER USER 'cdc'@'%' IDENTIFIED WITH mysql_native_password BY 'cdcpass';
GRANT ALL PRIVILEGES ON *.* TO 'cdc'@'%' WITH GRANT OPTION;
GRANT RELOAD, REPLICATION SLAVE, REPLICATION CLIENT, BINLOG_ADMIN ON *.* TO 'cdc'@'%';
FLUSH PRIVILEGES;
SQL
exec "$@"
EOF
RUN chmod +x /usr/local/bin/mysql-entrypoint.sh

ENV MYSQL_HOST=127.0.0.1
ENV MYSQL_PORT=3306
ENV MYSQL_USER=cdc
ENV MYSQL_PASSWORD=cdcpass

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + local MySQL (row-based binlog) environment for a MySQL binlog CDC indexer task."

ENTRYPOINT ["/usr/local/bin/mysql-entrypoint.sh"]
CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - MySQL 8 server + client (apt), configured for ROW binlog with FULL row image/metadata
#   - pip3 packages: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3,
#     pymysql==1.1.1, mysql-replication==1.0.16 (python-mysql-replication)
#   - ENTRYPOINT starts mysqld, provisions appdb.documents (empty) and the 'cdc' replication user
#   - /home/user/myproject: project directory for the candidate's solution

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
